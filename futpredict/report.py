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
                 models_info=None, home_adv_info=None):
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
        lstm_text = Text()
        if models_info.get("lstm"):
            l_d = models_info["lstm"]
            lstm_text.append(f"Win A: {l_d.get('win_a_pct', 0):.1f}%\nDraw:  {l_d.get('draw_pct', 0):.1f}%\nWin B: {l_d.get('win_b_pct', 0):.1f}%\n")
            if home_adv_info:
                lstm_text.append(f"Home Adv: {home_adv_info.get('factor', 0)*100:.1f}%\n({home_adv_info.get('confederation', 'Global')})", style="dim")
        
        grid.add_row(
            Panel(dc_text, title="[yellow]DIXON-COLES[/yellow]"),
            Panel(xgb_text, title="[green]XGBOOST[/green]"),
            Panel(lstm_text, title="[magenta]LSTM[/magenta]")
        )
        console.print(grid)

    # ── PREDICTIONS SECTION ──
    n_sims = outcomes.get("n_sims", 100000)
    console.print(f"\n[bold underline]MATCH OUTCOME[/bold underline] [dim]({n_sims:,} simulations)[/dim]")
    
    lstm_o = outcomes.get("lstm", outcomes.get("dc", {}))
    win_a = lstm_o.get("win_a_pct", 0)
    draw = lstm_o.get("draw_pct", 0)
    win_b = lstm_o.get("win_b_pct", 0)
    
    table_1x2 = Table(box=box.SIMPLE, show_header=False, expand=True)
    table_1x2.add_column("Outcome", style="bold")
    table_1x2.add_column("Prob", justify="right")
    table_1x2.add_column("Bar")
    
    table_1x2.add_row(f"{name_a} Win", color_pct_text(win_a), make_bar(win_a, color="green"))
    table_1x2.add_row("Draw", color_pct_text(draw), make_bar(draw, color="yellow"))
    table_1x2.add_row(f"{name_b} Win", color_pct_text(win_b), make_bar(win_b, color="red"))
    console.print(table_1x2)

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

    best_totals_diff = 0
    normal_totals_str = "NO BET"
    for thresh in ["1_5", "2_5", "3_5"]:
        o_val = xgb_o.get(f'over_{thresh}_pct', 0)
        u_val = xgb_o.get(f'under_{thresh}_pct', 0)
        if o_val > 61.5 and (o_val - 50.0) > best_totals_diff:
            best_totals_diff = o_val - 50.0
            normal_totals_str = f"Over {thresh.replace('_', '.')} Goals (STRONG SIGNAL: {color_pct_text(o_val)})"
        elif u_val > 58.0 and (u_val - 50.0) > best_totals_diff:
            best_totals_diff = u_val - 50.0
            normal_totals_str = f"Under {thresh.replace('_', '.')} Goals (STRONG SIGNAL: {color_pct_text(u_val)})"
            
    if best_totals_diff == 0:
        o35 = xgb_o.get('over_3_5_pct', 0)
        u35 = xgb_o.get('under_3_5_pct', 0)
        normal_totals_str = f"NO BET (Signal too weak: Over {color_pct_text(o35)} / Under {color_pct_text(u35)})"
    
    pts_diff = abs(fifa_pts_a - fifa_pts_b) if fifa_pts_a and fifa_pts_b else 0
    
    totals_final_str = normal_totals_str
    if pts_diff > 200:
        is_a_underdog = fifa_pts_a < fifa_pts_b
        underdog_form = form_a if is_a_underdog else form_b
        underdog_ga = underdog_form.get('weighted_ga', 1.5) if underdog_form else 1.5
        
        if underdog_ga < 1.2:
            u35 = xgb_o.get('under_3_5_pct', 0)
            u25 = xgb_o.get('under_2_5_pct', 0)

            if u35 > 65.0:
                totals_final_str = f"[bold cyan]Under 3.5 Goals (TOURNAMENT OVERRIDE - Elite Def: {color_pct_text(u35)})[/bold cyan]"
            else:
                totals_final_str = f"[bold cyan]Under 2.5 Goals (TOURNAMENT OVERRIDE - Elite Def: {color_pct_text(u25)})[/bold cyan]"
        else:
            o25 = xgb_o.get('over_2_5_pct', 0)
            o35 = xgb_o.get('over_3_5_pct', 0)
            ratio = o35 / o25 if o25 > 0 else 0
            if ratio > 0.65:
                totals_final_str = f"[bold red]Over 3.5 Goals (TOURNAMENT OVERRIDE - Weak Def: {color_pct_text(o35)})[/bold red]"
            else:
                totals_final_str = f"[bold yellow]Over 2.5 Goals (TOURNAMENT OVERRIDE - Weak Def: {color_pct_text(o25)})[/bold yellow]"
        totals_final_str += f"\n[dim]↳ Normal: {normal_totals_str} (Ignored due to >200pt mismatch)[/dim]"

    dc_1x = win_a + draw
    dc_x2 = win_b + draw
    dc_12 = win_a + win_b
    
    match_final_str = "NO BET (Highly unpredictable outcome)"
    
    WIN_THRESHOLD = 68.0
    DC_THRESHOLD = 70.0
    
    if win_a > WIN_THRESHOLD:
        match_final_str = f"[bold green]Direct Win {name_a}[/bold green] ({color_pct_text(win_a)})"
    elif win_b > WIN_THRESHOLD:
        match_final_str = f"[bold green]Direct Win {name_b}[/bold green] ({color_pct_text(win_b)})"
    else:
        if dc_1x > DC_THRESHOLD:
            match_final_str = f"[bold blue]Double Chance {name_a} or Draw[/bold blue] ({color_pct_text(dc_1x)})"
        elif dc_x2 > DC_THRESHOLD:
            match_final_str = f"[bold blue]Double Chance {name_b} or Draw[/bold blue] ({color_pct_text(dc_x2)})"
        elif dc_12 > DC_THRESHOLD:
            match_final_str = f"[bold blue]Double Chance {name_a} or {name_b}[/bold blue] ({color_pct_text(dc_12)})"

    console.print()
    advice_text = Text.from_markup(f"[bold]1X2 MATCH:[/bold] {match_final_str}\n[bold]TOTALS:[/bold] {totals_final_str}")
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
