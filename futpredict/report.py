"""
Console output formatting for the prediction report using rich.
"""
import time
from collections import Counter
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.text import Text
from rich.align import Align
from rich import box
import numpy as np
import scipy.stats as stats

console = Console()

def make_bar(pct, width=20, color="blue"):
    """Create a visual bar from a percentage, colored via rich tags."""
    filled = int(pct / 100.0 * width)
    bar_str = "█" * filled + "░" * (width - filled)
    return f"[{color}]{bar_str}[/{color}]"

def color_pct_text(pct):
    """Format a percentage with a color gradient."""
    if pct >= 80: color = "bold green"
    elif pct >= 60: color = "green"
    elif pct >= 40: color = "yellow"
    elif pct >= 20: color = "orange3"
    else: color = "red"
    return f"[{color}]{pct:5.1f}%[/{color}]"

def fmt_score(ga, gb, name_a, name_b):
    if ga > gb:
        return f"{ga}-{gb} ({name_a})"
    elif gb > ga:
        return f"{ga}-{gb} ({name_b})"
    else:
        return f"{ga}-{gb} (Draw)"

def print_report(name_a, name_b, db_name_a, db_name_b,
                 fifa_pts_a, fifa_pts_b, rank_a, rank_b,
                 form_a, form_b, h2h, outcomes, match_count,
                 models_info=None, home_adv_info=None, odds=None,
                 is_knockout=False, odds_q=None):
    """Print the comprehensive prediction report using rich."""
    
    # ── HEADER ──
    venue_str = models_info.get("venue", "neutral").replace("_", " ").title() if models_info else "Neutral"
    header_text = f"[bold cyan]FUTPREDICT v2.0[/bold cyan] — MATCH PREDICTION ENGINE\n[dim]Database: {match_count:,} records | Venue: {venue_str}[/dim]"
    console.print()
    console.print(Panel(Align.center(header_text), box=box.DOUBLE, border_style="cyan"))

    # ── TEAMS OVERVIEW (Columns) ──
    def create_team_panel(name, rank, pts, form, h2h_wins, h2h_xg):
        text = Text()
        text.append("FIFA RANKING: ", style="bold")
        text.append(f"#{rank} " if rank else "N/A ", style="yellow")
        text.append(f"({pts:.2f} pts)\n" if pts else "(N/A pts)\n", style="dim")
        
        if form:
            w, d, l = form["raw_wins"], form["raw_draws"], form["raw_losses"]
            form_str = "".join(r["result"][0] for r in form["recent_results"])
            text.append("FORM (LAST 30): ", style="bold")
            text.append(f"W{w} D{d} L{l} \n")
            text.append("   " + form_str + "\n", style="cyan")
            text.append(f"   GF {form['weighted_gf']:.2f}/g | GA {form['weighted_ga']:.2f}/g\n", style="dim")
            text.append(f"   Opponent Quality: {form['avg_opponent_strength']:.2f}\n", style="dim")
        
        text.append("H2H STATS: ", style="bold")
        text.append(f"{h2h_wins} wins\n")
        if h2h_xg is not None:
            text.append(f"   Avg Scoring: {h2h_xg:.2f}/g", style="dim")
            
        return Panel(text, title=f"[bold]{name}[/bold]", border_style="blue", expand=True)

    panel_a = create_team_panel(name_a, rank_a, fifa_pts_a, form_a, h2h['wins_a'], h2h.get('h2h_lambda_a'))
    panel_b = create_team_panel(name_b, rank_b, fifa_pts_b, form_b, h2h['wins_b'], h2h.get('h2h_lambda_b'))
    console.print(Columns([panel_a, panel_b], equal=True))
    
    # Delta
    if fifa_pts_a and fifa_pts_b:
        delta = abs(fifa_pts_a - fifa_pts_b)
        favor = name_a if fifa_pts_a > fifa_pts_b else name_b
        console.print(f"[bold magenta]   Δ[/bold magenta] {delta:.2f} pts advantage → [bold]{favor}[/bold]\n")

    # ── MODELS ARCHITECTURE ──
    if models_info:
        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_column()
        grid.add_column()
        grid.add_column()
        
        # DC
        dc_text = Text()
        if models_info.get("dc"):
            dc = models_info["dc"]
            dc_text.append(f"λ({name_a[:3]}) = {dc['lambda_h']:.3f}\nλ({name_b[:3]}) = {dc['lambda_a']:.3f}\n")
            dc_text.append(f"ρ = {dc.get('rho', 0):.4f}\nα = {dc.get('alpha', 0):.4f}", style="dim")
        
        # XGB
        xgb_text = Text()
        if models_info.get("xgb"):
            xgb_d = models_info["xgb"]
            xgb_text.append(f"Over 3.5: {xgb_d.get('over_3_5_pct', 0):.1f}%\nBTTS Yes: {xgb_d.get('btts_yes_pct', 0):.1f}%\n")
            if models_info.get("xgb_meta"):
                x_m = models_info["xgb_meta"]
                xgb_text.append(f"LogLoss T: {x_m.get('totals_logloss', 0):.3f}\nLogLoss B: {x_m.get('btts_logloss', 0):.3f}", style="dim")
                
        # LSTM
        lstm_agg_text = Text()
        if models_info.get("lstm_agg"):
            l_a = models_info["lstm_agg"]
            lstm_agg_text.append(f"Win A: {l_a.get('win_a_pct', 0):.1f}%\nDraw:  {l_a.get('draw_pct', 0):.1f}%\nWin B: {l_a.get('win_b_pct', 0):.1f}%")
            if home_adv_info:
                lstm_agg_text.append(f"\nHome Adv: {home_adv_info.get('factor', 0)*100:.1f}%\n({home_adv_info.get('confederation', 'Global')})", style="dim")
                
        lstm_cons_text = Text()
        if models_info.get("lstm_cons"):
            l_c = models_info["lstm_cons"]
            lstm_cons_text.append(f"Win A: {l_c.get('win_a_pct', 0):.1f}%\nDraw:  {l_c.get('draw_pct', 0):.1f}%\nWin B: {l_c.get('win_b_pct', 0):.1f}%")
            if home_adv_info:
                lstm_cons_text.append(f"\nHome Adv: {home_adv_info.get('factor', 0)*100:.1f}%\n({home_adv_info.get('confederation', 'Global')})", style="dim")
        
        grid.add_row(
            Panel(dc_text, title="[yellow]DIXON-COLES[/yellow]"),
            Panel(xgb_text, title="[green]XGBOOST[/green]"),
            Panel(lstm_agg_text, title="[magenta]LSTM AGG[/magenta]"),
            Panel(lstm_cons_text, title="[magenta]LSTM CONS[/magenta]")
        )
        console.print(grid)

    # ── PREDICTIONS SECTION ──
    n_sims = outcomes.get("n_sims", 100000)
    
    lstm_agg = outcomes.get("lstm_agg", {})
    lstm_cons = outcomes.get("lstm_cons", {})
    
    console.print(f"\n[bold underline]MATCH OUTCOME (Aggressive Focal)[/bold underline] [dim]({n_sims:,} simulations)[/dim]")
    l_a_o = lstm_agg if lstm_agg else outcomes.get("dc", {})
    w_a_agg = l_a_o.get("win_a_pct", 0)
    d_agg = l_a_o.get("draw_pct", 0)
    w_b_agg = l_a_o.get("win_b_pct", 0)
    
    table_agg = Table(box=box.SIMPLE, show_header=False, expand=True)
    table_agg.add_column("Outcome", style="bold")
    table_agg.add_column("Prob", justify="right")
    table_agg.add_column("Bar")
    table_agg.add_row(f"{name_a} Win", color_pct_text(w_a_agg), make_bar(w_a_agg, color="green"))
    table_agg.add_row("Draw", color_pct_text(d_agg), make_bar(d_agg, color="yellow"))
    table_agg.add_row(f"{name_b} Win", color_pct_text(w_b_agg), make_bar(w_b_agg, color="red"))
    console.print(table_agg)
    
    console.print(f"\n[bold underline]MATCH OUTCOME (Conservative Cross-Entropy)[/bold underline] [dim]({n_sims:,} simulations)[/dim]")
    l_c_o = lstm_cons if lstm_cons else outcomes.get("dc", {})
    w_a_cons = l_c_o.get("win_a_pct", 0)
    d_cons = l_c_o.get("draw_pct", 0)
    w_b_cons = l_c_o.get("win_b_pct", 0)
    
    table_cons = Table(box=box.SIMPLE, show_header=False, expand=True)
    table_cons.add_column("Outcome", style="bold")
    table_cons.add_column("Prob", justify="right")
    table_cons.add_column("Bar")
    table_cons.add_row(f"{name_a} Win", color_pct_text(w_a_cons), make_bar(w_a_cons, color="green"))
    table_cons.add_row("Draw", color_pct_text(d_cons), make_bar(d_cons, color="yellow"))
    table_cons.add_row(f"{name_b} Win", color_pct_text(w_b_cons), make_bar(w_b_cons, color="red"))
    console.print(table_cons)

    w_a_mean = (w_a_agg + w_a_cons) / 2
    d_mean = (d_agg + d_cons) / 2
    w_b_mean = (w_b_agg + w_b_cons) / 2
    
    console.print(f"\n[bold underline]MATCH OUTCOME (Consensus Mean)[/bold underline] [dim]({n_sims:,} simulations)[/dim]")
    table_mean = Table(box=box.SIMPLE, show_header=False, expand=True)
    table_mean.add_column("Outcome", style="bold")
    table_mean.add_column("Prob", justify="right")
    table_mean.add_column("Bar")
    table_mean.add_row(f"{name_a} Win", color_pct_text(w_a_mean), make_bar(w_a_mean, color="green"))
    table_mean.add_row("Draw", color_pct_text(d_mean), make_bar(d_mean, color="yellow"))
    table_mean.add_row(f"{name_b} Win", color_pct_text(w_b_mean), make_bar(w_b_mean, color="red"))
    console.print(table_mean)
    
    q_a = 0
    q_b = 0
    if is_knockout:
        elo_a = fifa_pts_a if fifa_pts_a else 1500
        elo_b = fifa_pts_b if fifa_pts_b else 1500
        total_elo = elo_a + elo_b
        edge_a = elo_a / total_elo if total_elo > 0 else 0.5
        edge_b = 1.0 - edge_a
        
        q_a = w_a_mean + (d_mean * edge_a)
        q_b = w_b_mean + (d_mean * edge_b)
        
        console.print(f"\n[bold underline]TO QUALIFY / ADVANCE[/bold underline] [dim](Includes Extra Time & Penalties)[/dim]")
        table_q = Table(box=box.SIMPLE, show_header=False, expand=True)
        table_q.add_column("Outcome", style="bold")
        table_q.add_column("Prob", justify="right")
        table_q.add_column("Bar")
        table_q.add_row(f"{name_a} Advances", color_pct_text(q_a), make_bar(q_a, color="cyan"))
        table_q.add_row(f"{name_b} Advances", color_pct_text(q_b), make_bar(q_b, color="magenta"))
        
        xgb_info = models_info.get("xgb", {})
        et_prob_val = xgb_info.get("xgb_et_prob")
        if et_prob_val is not None:
            et_pct = et_prob_val * 100
            reg_pct = 100.0 - et_pct
            table_q.add_row("", "", "")
            table_q.add_row("↳ Decided in 90 Minutes", color_pct_text(reg_pct), make_bar(reg_pct, color="white"))
            table_q.add_row("↳ Goes to Extra Time / Pens", color_pct_text(et_pct), make_bar(et_pct, color="yellow"))
            
        console.print(table_q)

    # Top Scorelines
    dc_o = outcomes.get("dc", outcomes.get("xgb", {}))
    if dc_o and "top_scores" in dc_o:
        console.print("\n[bold underline]TOP SCORELINES[/bold underline] [dim](Dixon-Coles)[/dim]")
        t_score = Table(box=box.SIMPLE, show_header=False, expand=True)
        t_score.add_column("Rank", style="dim")
        t_score.add_column("Scoreline", style="bold")
        t_score.add_column("Prob", justify="right")
        t_score.add_column("Bar")
        
        for i, ((ga, gb), count) in enumerate(dc_o.get("top_scores", [])[:5]):
            pct = (count / n_sims) * 100
            label = fmt_score(ga, gb, name_a, name_b)
            t_score.add_row(f"{i+1}.", label, color_pct_text(pct), make_bar(pct, width=30, color="cyan"))
        console.print(t_score)

    # Markets
    xgb_o = outcomes.get("xgb", outcomes.get("lstm", {}))
    if xgb_o:
        console.print("\n[bold underline]MARKET PROBABILITIES[/bold underline] [dim](XGBoost)[/dim]")
        t_mark = Table(box=box.SIMPLE, show_header=False, expand=True)
        t_mark.add_column()
        t_mark.add_column()
        t_mark.add_column()
        t_mark.add_column()
        
        t_mark.add_row("Over 0.5:", color_pct_text(xgb_o.get('over_0_5_pct', 0)), "Under 0.5:", color_pct_text(xgb_o.get('under_0_5_pct', 0)))
        t_mark.add_row("Over 1.5:", color_pct_text(xgb_o.get('over_1_5_pct', 0)), "Under 1.5:", color_pct_text(xgb_o.get('under_1_5_pct', 0)))
        t_mark.add_row("Over 2.5:", color_pct_text(xgb_o.get('over_2_5_pct', 0)), "Under 2.5:", color_pct_text(xgb_o.get('under_2_5_pct', 0)))
        t_mark.add_row("Over 3.5:", color_pct_text(xgb_o.get('over_3_5_pct', 0)), "Under 3.5:", color_pct_text(xgb_o.get('under_3_5_pct', 0)))
        t_mark.add_row("Over 4.5:", color_pct_text(xgb_o.get('over_4_5_pct', 0)), "Under 4.5:", color_pct_text(xgb_o.get('under_4_5_pct', 0)))
        t_mark.add_row("Over 5.5:", color_pct_text(xgb_o.get('over_5_5_pct', 0)), "Under 5.5:", color_pct_text(xgb_o.get('under_5_5_pct', 0)))
        t_mark.add_row("", "", "", "")
        t_mark.add_row("BTTS Yes:", color_pct_text(xgb_o.get('btts_yes_pct', 0)), "BTTS No:", color_pct_text(xgb_o.get('btts_no_pct', 0)))
        console.print(t_mark)
        

    # ── GOALSCORERS ──
    goalscorers = models_info.get("goalscorers", [])
    if goalscorers:
        console.print("\n[bold underline]ANYTIME GOALSCORER PREDICTION[/bold underline] [dim](XGBoost Player Models)[/dim]")
        
        t_scorers = Table(box=box.SIMPLE, show_header=True, expand=True)
        t_scorers.add_column("Rank", style="dim")
        t_scorers.add_column("Player", style="bold")
        t_scorers.add_column("Team", style="dim")
        t_scorers.add_column("Pos", style="dim")
        t_scorers.add_column("Avg xG", justify="right")
        t_scorers.add_column("Prob", justify="right", style="bold green")
        
        for i, p in enumerate(goalscorers[:5]):
            t_scorers.add_row(
                f"{i+1}.",
                p["player_name"],
                p["team"],
                p["position"],
                f"{p['xg_avg']:.2f}",
                color_pct_text(p["prob"])
            )
        console.print(t_scorers)

    # ── ADVANCED METRICS ──
    adv = models_info.get("advanced")
    if adv:
        console.print("\n[bold underline]ADVANCED METRICS PREDICTION[/bold underline] [dim](Time-Binned Deep Features)[/dim]")
        
        t_adv = Table(box=box.SIMPLE, show_header=True, expand=True)
        t_adv.add_column("Corners", justify="left")
        t_adv.add_column("", justify="left")
        t_adv.add_column("Cards", justify="left")
        t_adv.add_column("", justify="left")
        t_adv.add_column("Target Shots", justify="left")
        t_adv.add_column("", justify="left")
        
        cor = adv["expected_corners"]
        car = adv["expected_cards"]
        sot = adv["expected_sot"]
        pos = adv["home_possession"]
        
        base_cor = max(0, int(round(cor)) - 2)
        base_car = max(0, int(round(car)) - 2)
        base_sot = max(0, int(round(sot)) - 2)
        
        for i in range(5):
            c_val = base_cor + i
            ca_val = base_car + i
            s_val = base_sot + i
            
            o_cor = (1 - stats.poisson.cdf(c_val, cor)) * 100
            u_cor = stats.poisson.cdf(c_val, cor) * 100
            
            o_car = (1 - stats.poisson.cdf(ca_val, car)) * 100
            u_car = stats.poisson.cdf(ca_val, car) * 100
            
            o_sot = (1 - stats.poisson.cdf(s_val, sot)) * 100
            u_sot = stats.poisson.cdf(s_val, sot) * 100
            
            t_adv.add_row(
                f"Over {c_val}.5:  {color_pct_text(o_cor)}", f"Under {c_val}.5:  {color_pct_text(u_cor)}",
                f"Over {ca_val}.5:  {color_pct_text(o_car)}", f"Under {ca_val}.5:  {color_pct_text(u_car)}",
                f"Over {s_val}.5:  {color_pct_text(o_sot)}", f"Under {s_val}.5:  {color_pct_text(u_sot)}"
            )
            
        console.print(t_adv)
        
        pos_h = min(100.0, max(0.0, pos))
        pos_a = 100.0 - pos_h
        console.print(f"  [bold magenta]Possession:[/bold magenta] {name_a}: [bold]{pos_h:.1f}%[/bold] │ {name_b}: [bold]{pos_a:.1f}%[/bold]\n")

    # ── Totals Betting: Dynamic Line Shifting (Target 78%) ──
    TARGET_WIN_RATE = 78.0
    normal_totals_str = "NO BET"
    
    over_lines = ["5_5", "4_5", "3_5", "2_5", "1_5"]
    under_lines = ["1_5", "2_5", "3_5", "4_5", "5_5"]
    
    best_over = None
    for thresh in over_lines:
        o_val = xgb_o.get(f"over_{thresh}_pct", 0)
        if o_val > TARGET_WIN_RATE:
            best_over = (thresh, o_val)
            break
            
    best_under = None
    for thresh in under_lines:
        u_val = xgb_o.get(f"under_{thresh}_pct", 0)
        if u_val > TARGET_WIN_RATE:
            best_under = (thresh, u_val)
            break
            
    def dist(t):
        return abs(float(t.replace('_', '.')) - 2.5)
        
    if best_over and best_under:
        do = dist(best_over[0])
        du = dist(best_under[0])
        if do < du:
            normal_totals_str = f"Over {best_over[0].replace('_', '.')} Goals (High Probability Target: {color_pct_text(best_over[1])})"
        elif du < do:
            normal_totals_str = f"Under {best_under[0].replace('_', '.')} Goals (High Probability Target: {color_pct_text(best_under[1])})"
        else:
            if best_over[1] >= best_under[1]:
                normal_totals_str = f"Over {best_over[0].replace('_', '.')} Goals (High Probability Target: {color_pct_text(best_over[1])})"
            else:
                normal_totals_str = f"Under {best_under[0].replace('_', '.')} Goals (High Probability Target: {color_pct_text(best_under[1])})"
    elif best_over:
        normal_totals_str = f"Over {best_over[0].replace('_', '.')} Goals (High Probability Target: {color_pct_text(best_over[1])})"
    elif best_under:
        normal_totals_str = f"Under {best_under[0].replace('_', '.')} Goals (High Probability Target: {color_pct_text(best_under[1])})"
        
    if normal_totals_str == "NO BET":
        o35 = xgb_o.get('over_3_5_pct', 0)
        u35 = xgb_o.get('under_3_5_pct', 0)
        normal_totals_str = f"NO BET (No line meets 78% target: Over 3.5 {color_pct_text(o35)} / Under 3.5 {color_pct_text(u35)})"
    
    pts_diff = abs(fifa_pts_a - fifa_pts_b) if fifa_pts_a and fifa_pts_b else 0
    
    totals_final_str = normal_totals_str
    if pts_diff > 200:
        x_o35 = xgb_o.get('over_3_5_pct', 0)
        x_u25 = xgb_o.get('under_2_5_pct', 0)
        
        lstm_agg = models_info.get("lstm_agg") or {}
        adv_details = models_info.get("advanced") or {}
        
        dl_eg = lstm_agg.get("expected_goals", 2.5)
        adv_sot = adv_details.get("expected_sot", 8.5)
        
        if x_o35 > 40.0 and dl_eg > 2.8 and adv_sot > 8.5:
            totals_final_str = f"[bold red]Over 3.5 Goals (SMART TOURNAMENT OVERRIDE - DL/SOT Fusion: {color_pct_text(x_o35)})[/bold red]"
        elif x_u25 > 40.0 and dl_eg < 2.0 and adv_sot < 7.5:
            totals_final_str = f"[bold cyan]Under 2.5 Goals (SMART TOURNAMENT OVERRIDE - DL/SOT Fusion: {color_pct_text(x_u25)})[/bold cyan]"
        else:
            # Fallback for ambiguous mismatch
            o15 = xgb_o.get('over_1_5_pct', 0)
            o25 = xgb_o.get('over_2_5_pct', 0)
            u25 = xgb_o.get('under_2_5_pct', 0)
            if o25 > 58.0:
                totals_final_str = f"[bold yellow]Over 2.5 Goals (SMART TOURNAMENT OVERRIDE: {color_pct_text(o25)})[/bold yellow]"
            elif u25 > 58.0:
                totals_final_str = f"[bold cyan]Under 2.5 Goals (SMART TOURNAMENT OVERRIDE: {color_pct_text(u25)})[/bold cyan]"
            elif o15 > 75.0:
                totals_final_str = f"[bold green]Over 1.5 Goals (SMART TOURNAMENT SAFE: {color_pct_text(o15)})[/bold green]"
            else:
                totals_final_str = normal_totals_str
                
        if totals_final_str != normal_totals_str:
            totals_final_str += f"\n[dim]↳ Normal: {normal_totals_str} (Ignored due to >200pt mismatch)[/dim]"

    WIN_THRESHOLD = 68.0
    DC_THRESHOLD = 70.0
    
    def _generate_1x2_advice(lstm_details, odds, tag):
        w_a = lstm_details.get("win_a_pct", 0)
        d = lstm_details.get("draw_pct", 0)
        w_b = lstm_details.get("win_b_pct", 0)
        
        d_1x = w_a + d
        d_x2 = w_b + d
        d_12 = w_a + w_b
        
        match_final_str = "NO BET (Highly unpredictable outcome)"
        raw_prediction = "NO_BET"
        
        if odds is not None:
            odds_a, odds_d, odds_b = odds
            
            def _compute_ev(model_prob, dec_odds):
                if dec_odds is None or dec_odds <= 1.0: return -100.0
                return ((model_prob / 100.0) * dec_odds) - 1.0
                
            def _implied_prob(dec_odds):
                if dec_odds is None or dec_odds <= 1.0: return 100.0
                return (1.0 / dec_odds) * 100.0
                
            markets = []
            if odds_a: markets.append(("1", name_a, w_a, odds_a))
            if odds_d: markets.append(("X", "Draw", d, odds_d))
            if odds_b: markets.append(("2", name_b, w_b, odds_b))
            
            best_ev = 0.0
            best_market = None
            
            for m_key, m_name, m_prob, m_odds in markets:
                ev = _compute_ev(m_prob, m_odds)
                if ev > best_ev:
                    best_ev = ev
                    best_market = (m_key, m_name, m_prob, m_odds, ev)
                    
            if best_market and best_ev > 0.0:
                m_key, m_name, m_prob, m_odds, ev = best_market
                imp_prob = _implied_prob(m_odds)
                raw_prediction = m_key
                match_final_str = f"[bold green]{m_name}[/bold green] (Model: {color_pct_text(m_prob)} vs Market: {imp_prob:.1f}%) → EV [bold green]+{ev*100:.1f}%[/bold green] ✓ VALUE BET"
            elif best_market:
                m_key, m_name, m_prob, m_odds, ev = best_market
                raw_prediction = "NO_BET"
                imp_prob = _implied_prob(m_odds)
                match_final_str = f"[bold red]NO VALUE[/bold red] (Best option {m_name}: Model {color_pct_text(m_prob)} < Market {imp_prob:.1f}%)"
            else:
                raw_prediction = "NO_BET"
                match_final_str = "NO BET (No valid odds provided for EV calculation)"
        else:
            if w_a > WIN_THRESHOLD:
                raw_prediction = "1"
                match_final_str = f"[bold green]Direct Win {name_a}[/bold green] ({color_pct_text(w_a)})"
            elif w_b > WIN_THRESHOLD:
                raw_prediction = "2"
                match_final_str = f"[bold green]Direct Win {name_b}[/bold green] ({color_pct_text(w_b)})"
            else:
                if d_1x > DC_THRESHOLD:
                    raw_prediction = "1X"
                    match_final_str = f"[bold blue]Double Chance {name_a} or Draw[/bold blue] ({color_pct_text(d_1x)})"
                elif d_x2 > DC_THRESHOLD:
                    raw_prediction = "X2"
                    match_final_str = f"[bold blue]Double Chance {name_b} or Draw[/bold blue] ({color_pct_text(d_x2)})"
                elif d_12 > DC_THRESHOLD:
                    raw_prediction = "12"
                    match_final_str = f"[bold blue]Double Chance {name_a} or {name_b}[/bold blue] ({color_pct_text(d_12)})"
            match_final_str += " [dim]⚠ NO ODDS PROVIDED[/dim]"
            
        return f"[bold]1X2 {tag}:[/bold] {match_final_str}", raw_prediction
        
    def _extract_bet_name(formatted_str):
        # Extract the bet name from inside the [bold color] tags
        import re
        match = re.search(r'\[bold [^\]]+\](.*?)\[/bold [^\]]+\]', formatted_str)
        if match:
            return match.group(1)
        return formatted_str

    agg_advice, raw_agg = _generate_1x2_advice(lstm_agg, odds, "Aggressive (Focal)")
    cons_advice, raw_cons = _generate_1x2_advice(lstm_cons, odds, "Conservative (Cross-Ent)")
    
    agg_bet_name = _extract_bet_name(agg_advice)
    cons_bet_name = _extract_bet_name(cons_advice)

    consensus_str = "[bold red]NO BET[/bold red] (Models Conflict)"
    if raw_agg == "NO_BET" or raw_cons == "NO_BET":
        consensus_str = "[bold red]NO BET[/bold red] (Unpredictable)"
    elif raw_agg == raw_cons:
        consensus_str = f"[bold green]CONFIRMED MATCH[/bold green] ([bold cyan]{agg_bet_name}[/bold cyan])"
    elif raw_agg in raw_cons:
        consensus_str = f"[bold yellow]PARTIAL MATCH[/bold yellow] ([bold cyan]{cons_bet_name}[/bold cyan])"
    elif raw_cons in raw_agg:
        consensus_str = f"[bold yellow]PARTIAL MATCH[/bold yellow] ([bold cyan]{agg_bet_name}[/bold cyan])"

    if is_knockout:
        if q_a > q_b:
            consensus_str = f"[bold green]KNOCKOUT ADVANCE[/bold green] ([bold cyan]Direct Win {name_a}:[/bold cyan] {color_pct_text(q_a)})"
        else:
            consensus_str = f"[bold green]KNOCKOUT ADVANCE[/bold green] ([bold cyan]Direct Win {name_b}:[/bold cyan] {color_pct_text(q_b)})"

        xgb_info = models_info.get("xgb", {})
        et_prob_val = xgb_info.get("xgb_et_prob")
        if et_prob_val is not None:
            et_pct = et_prob_val * 100
            reg_pct = 100.0 - et_pct
            consensus_str += f"\n         [dim]↳ Regular time[/dim] {color_pct_text(reg_pct)}\n         [dim]↳ Extra Time[/dim] {color_pct_text(et_pct)}"

    qualify_str = ""
    if is_knockout:
        if odds_q and (odds_q[0] is not None or odds_q[1] is not None):
            odds_qa, odds_qb = odds_q
            ev_a = ((q_a / 100.0) * odds_qa) - 1.0 if odds_qa else -100.0
            ev_b = ((q_b / 100.0) * odds_qb) - 1.0 if odds_qb else -100.0
            
            if ev_a > 0 and ev_a >= ev_b:
                imp_prob = (1.0 / odds_qa) * 100.0
                qualify_str = f"\n[bold cyan]QUALIFY:[/bold cyan] [bold green]{name_a} Advances[/bold green] (Model: {color_pct_text(q_a)} vs Market: {imp_prob:.1f}%) → EV [bold green]+{ev_a*100:.1f}%[/bold green] ✓ VALUE BET"
            elif ev_b > 0 and ev_b > ev_a:
                imp_prob = (1.0 / odds_qb) * 100.0
                qualify_str = f"\n[bold cyan]QUALIFY:[/bold cyan] [bold green]{name_b} Advances[/bold green] (Model: {color_pct_text(q_b)} vs Market: {imp_prob:.1f}%) → EV [bold green]+{ev_b*100:.1f}%[/bold green] ✓ VALUE BET"
            else:
                qualify_str = f"\n[bold cyan]QUALIFY:[/bold cyan] [bold red]NO VALUE[/bold red] (No value found on To Qualify odds)"
        else:
            if q_a > 68.0:
                qualify_str = f"\n[bold cyan]QUALIFY:[/bold cyan] [bold green]{name_a} Advances[/bold green] ({color_pct_text(q_a)}) [dim]⚠ NO ODDS PROVIDED[/dim]"
            elif q_b > 68.0:
                qualify_str = f"\n[bold cyan]QUALIFY:[/bold cyan] [bold green]{name_b} Advances[/bold green] ({color_pct_text(q_b)}) [dim]⚠ NO ODDS PROVIDED[/dim]"

    console.print()
    advice_text = Text.from_markup(f"{agg_advice}\n{cons_advice}\n[bold]1X2 CONSENSUS:[/bold] {consensus_str}{qualify_str}\n[bold]TOTALS:[/bold] {totals_final_str}")
    console.print(Panel(advice_text, title="[bold]BETTING ADVICE (VALUE & CONFIDENCE)[/bold]", border_style="green"))
    # ── RECENT RESULTS ──
    def build_recent_results_text(form):
        if not form or not form.get("recent_results"):
            return "No data."
        t = Text()
        for r in form["recent_results"][:5]: # Show top 5 to save space
            if r["result"] == "W":
                res_tag = "[bold green]W[/bold green]"
            elif r["result"] == "D":
                res_tag = "[bold yellow]D[/bold yellow]"
            else:
                res_tag = "[bold red]L[/bold red]"
            t.append_text(Text.from_markup(f"{res_tag} {r['date']}  {r['gf']}-{r['ga']} vs {r['opponent'][:12]}\n"))
        return t

    console.print()
    res_a = Panel(build_recent_results_text(form_a), title=f"[dim]RECENT:[/dim] {name_a.upper()}", expand=True)
    res_b = Panel(build_recent_results_text(form_b), title=f"[dim]RECENT:[/dim] {name_b.upper()}", expand=True)
    console.print(Columns([res_a, res_b], equal=True))
    console.print()
