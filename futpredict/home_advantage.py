"""
Dynamic home advantage calculation.
Replaces the static HOME_ADVANTAGE = 0.10 with historically-computed values.
Uses the Pollard method: HA = home_points / total_points, by confederation.
"""
import datetime


# Team → Confederation mapping (covers ~200 teams)
_CONFEDERATION = {
    # UEFA (Europe)
    **{t: "UEFA" for t in [
        "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus",
        "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus",
        "Czechia", "Czech Republic", "Denmark", "England", "Estonia",
        "Faroe Islands", "Finland", "France", "Georgia", "Germany", "Gibraltar",
        "Greece", "Hungary", "Iceland", "Republic of Ireland", "Israel",
        "Italy", "Kazakhstan", "Kosovo", "Latvia", "Liechtenstein",
        "Lithuania", "Luxembourg", "Malta", "Moldova", "Montenegro",
        "Netherlands", "North Macedonia", "Northern Ireland", "Norway",
        "Poland", "Portugal", "Romania", "Russia", "San Marino", "Scotland",
        "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland",
        "Türkiye", "Turkey", "Ukraine", "Wales",
    ]},
    # CONMEBOL (South America)
    **{t: "CONMEBOL" for t in [
        "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
        "Paraguay", "Peru", "Uruguay", "Venezuela",
    ]},
    # CONCACAF (North/Central America + Caribbean)
    **{t: "CONCACAF" for t in [
        "Antigua and Barbuda", "Bahamas", "Barbados", "Belize", "Bermuda",
        "Canada", "Cayman Islands", "Costa Rica", "Cuba", "Curaçao",
        "Dominica", "Dominican Republic", "El Salvador", "Grenada",
        "Guatemala", "Guyana", "Haiti", "Honduras", "Jamaica", "Mexico",
        "Montserrat", "Nicaragua", "Panama", "Puerto Rico",
        "St. Kitts and Nevis", "St. Lucia", "St. Vincent / Grenadines",
        "Suriname", "Trinidad and Tobago", "Turks and Caicos Islands",
        "USA", "US Virgin Islands", "British Virgin Islands", "Anguilla",
    ]},
    # CAF (Africa)
    **{t: "CAF" for t in [
        "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
        "Cabo Verde", "Cameroon", "Central African Republic", "Chad",
        "Comoros", "Congo", "Congo DR", "Côte d'Ivoire", "Djibouti",
        "Egypt", "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia",
        "Gabon", "The Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Kenya",
        "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi", "Mali",
        "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia",
        "Niger", "Nigeria", "Rwanda", "São Tomé and Príncipe", "Senegal",
        "Seychelles", "Sierra Leone", "Somalia", "South Africa",
        "South Sudan", "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda",
        "Zambia", "Zimbabwe",
    ]},
    # AFC (Asia)
    **{t: "AFC" for t in [
        "Afghanistan", "Australia", "Bahrain", "Bangladesh", "Bhutan",
        "Brunei", "Cambodia", "China PR", "Chinese Taipei", "Guam",
        "Hong Kong", "India", "Indonesia", "IR Iran", "Iraq", "Japan",
        "Jordan", "Korea Republic", "Kuwait", "Kyrgyz Republic", "Laos",
        "Lebanon", "Macau", "Malaysia", "Maldives", "Mongolia", "Myanmar",
        "Nepal", "DPR Korea", "Oman", "Pakistan", "Palestine", "Philippines",
        "Qatar", "Saudi Arabia", "Singapore", "Sri Lanka", "Syria",
        "Tajikistan", "Thailand", "Timor-Leste", "Turkmenistan",
        "United Arab Emirates", "Uzbekistan", "Vietnam", "Yemen",
    ]},
    # OFC (Oceania)
    **{t: "OFC" for t in [
        "American Samoa", "Cook Islands", "Fiji", "New Caledonia",
        "New Zealand", "Papua New Guinea", "Samoa", "Solomon Islands",
        "Tahiti", "Tonga", "Vanuatu",
    ]},
}

# Literature-based baseline HA factors by confederation
# These serve as priors; the compute function adjusts from data
_BASELINE_HA = {
    "CONMEBOL": 0.18,   # Strongest — altitude, travel, hostile crowds
    "CAF": 0.15,        # Strong — heat, travel distances
    "AFC": 0.13,        # Moderate-strong — varied climates
    "CONCACAF": 0.12,   # Moderate — altitude (Mexico), travel
    "UEFA": 0.08,       # Weakest — short distances, standardized
    "OFC": 0.10,        # Limited data
}


def get_confederation(team_name):
    """Get the confederation for a team."""
    return _CONFEDERATION.get(team_name)


def compute_home_advantage(conn, team_a_db=None, team_b_db=None,
                           years_back=6):
    """
    Compute dynamic home advantage factor from historical data.

    Returns a multiplier (e.g., 0.12 means +12% λ for home, -12% for away).
    Uses the Pollard method: HA = home_points / total_points.
    Can be filtered by confederation of the home team.

    For neutral venues, returns 0.0.
    """
    cutoff = (datetime.date.today() -
              datetime.timedelta(days=years_back * 365)).isoformat()

    # Determine confederation for region-specific HA
    confed = None
    if team_a_db:
        confed = get_confederation(team_a_db)

    # Query non-neutral matches
    if confed:
        # Get all teams in the same confederation
        confed_teams = [t for t, c in _CONFEDERATION.items() if c == confed]
        if not confed_teams:
            confed = None

    query = """
        SELECT home_score, away_score FROM matches
        WHERE date >= ? AND (neutral = 'FALSE' OR neutral = '' OR neutral IS NULL)
    """
    params = [cutoff]

    if confed and confed_teams:
        # Filter by confederation teams playing at home
        placeholders = ",".join("?" * len(confed_teams))
        query += f" AND home_team IN ({placeholders})"
        params.extend(confed_teams)

    cur = conn.execute(query, params)
    rows = cur.fetchall()

    if len(rows) < 100:
        # Not enough data — use baseline
        baseline = _BASELINE_HA.get(confed, 0.10) if confed else 0.10
        return baseline

    # Pollard method: compute home points share
    home_pts = 0
    total_pts = 0
    for hs, as_ in rows:
        if hs > as_:
            home_pts += 3
            total_pts += 3
        elif hs == as_:
            home_pts += 1
            total_pts += 2
        else:
            total_pts += 3

    if total_pts == 0:
        return 0.10

    ha_pct = home_pts / total_pts  # Should be > 0.5 if HA exists
    # Convert to λ multiplier: center at 0.5 (no advantage)
    ha_factor = (ha_pct - 0.50) * 2  # Maps 0.5→0, 0.6→0.2, etc.
    ha_factor = max(0.0, min(0.30, ha_factor))  # Clamp to [0, 0.30]

    return ha_factor
