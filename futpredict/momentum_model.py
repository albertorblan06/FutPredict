import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from futpredict.config import DATA_DIR, TRAIN_END_DATE
from futpredict.xgb_advanced import _get_time_bins

MOMENTUM_MODEL_PATH = os.path.join(DATA_DIR, "momentum_ae.pt")

class MomentumTransformer(nn.Module):
    def __init__(self, embed_dim=8):
        super().__init__()
        # Input: (batch, seq=6, features=4)
        self.feature_proj = nn.Linear(4, 16)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.compress = nn.Sequential(
            nn.Flatten(), # 6 * 16 = 96
            nn.Linear(96, embed_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 24) # output back to 6 * 4
        )
        
    def forward(self, x):
        # x: (batch, 6, 4)
        x_proj = self.feature_proj(x)
        trans_out = self.transformer(x_proj)
        encoded = self.compress(trans_out)
        decoded = self.decoder(encoded)
        return encoded, decoded.view(-1, 6, 4)

def build_momentum_dataset(conn, train_end_date):
    """Extract all time-bins to pre-train the autoencoder."""
    query = f"""
        SELECT home_possession_bins, away_possession_bins,
               home_corners_bins, away_corners_bins,
               home_cards_bins, away_cards_bins,
               home_sot_bins, away_sot_bins
        FROM advanced_stats
        WHERE match_date < '{train_end_date}'
    """
    try:
        df = pd.read_sql_query(query, conn)
    except Exception:
        return []
        
    samples = []
    for _, row in df.iterrows():
        # Home sequence
        h_pos = np.array(_get_time_bins(row["home_possession_bins"], 50)) / 100.0
        h_cor = np.array(_get_time_bins(row["home_corners_bins"], 0))
        h_car = np.array(_get_time_bins(row["home_cards_bins"], 0))
        h_sot = np.array(_get_time_bins(row["home_sot_bins"], 0))
        h_seq = np.stack([h_pos, h_cor, h_car, h_sot], axis=1) # (6, 4)
        samples.append(h_seq)
        
        # Away sequence
        a_pos = np.array(_get_time_bins(row["away_possession_bins"], 50)) / 100.0
        a_cor = np.array(_get_time_bins(row["away_corners_bins"], 0))
        a_car = np.array(_get_time_bins(row["away_cards_bins"], 0))
        a_sot = np.array(_get_time_bins(row["away_sot_bins"], 0))
        a_seq = np.stack([a_pos, a_cor, a_car, a_sot], axis=1) # (6, 4)
        samples.append(a_seq)
        
    return samples

def train_momentum_model(conn, force=False):
    if not force and os.path.exists(MOMENTUM_MODEL_PATH):
        try:
            model = MomentumTransformer(embed_dim=8)
            model.load_state_dict(torch.load(MOMENTUM_MODEL_PATH))
            model.eval()
            return model
        except Exception:
            pass
        
    print("   ⬇  Building dataset for Momentum Transformer...")
    samples = build_momentum_dataset(conn, TRAIN_END_DATE)
    if len(samples) < 100:
        print("   ⚠ Not enough data for momentum training.")
        return None
        
    tensor_x = torch.tensor(np.array(samples), dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(tensor_x)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    model = MomentumTransformer(embed_dim=8)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    print(f"   ⚙  Training Momentum AutoEncoder ({len(samples)} samples)...")
    torch.manual_seed(42)
    np.random.seed(42)
    model.train()
    for epoch in range(15): # Fast pre-training
        epoch_loss = 0.0
        for batch in loader:
            x = batch[0]
            optimizer.zero_grad()
            _, decoded = model(x)
            loss = criterion(decoded, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
    torch.save(model.state_dict(), MOMENTUM_MODEL_PATH)
    model.eval()
    print("   ✓  Momentum AutoEncoder trained and cached.")
    return model

def get_momentum_vector(model, time_bins_dict):
    """
    time_bins_dict: {'pos': [6], 'cor': [6], 'car': [6], 'sot': [6]}
    Returns numpy array of shape (8,)
    """
    if model is None:
        return np.zeros(8)
        
    pos = np.array(time_bins_dict['pos']) / 100.0
    cor = np.array(time_bins_dict['cor'])
    car = np.array(time_bins_dict['car'])
    sot = np.array(time_bins_dict['sot'])
    
    seq = np.stack([pos, cor, car, sot], axis=1) # (6, 4)
    tensor_in = torch.tensor(seq, dtype=torch.float32).unsqueeze(0) # (1, 6, 4)
    
    with torch.no_grad():
        encoded, _ = model(tensor_in)
        
    return encoded.numpy()[0]
