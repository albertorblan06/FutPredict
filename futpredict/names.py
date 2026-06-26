"""
Team name normalization — maps between different naming conventions.
Handles English names, Spanish names, ISO codes, and dataset variants.
"""


_NAME_GROUPS = [
    ("United States", "USA", "US", "U.S.A.", "United States of America",
     "Estados Unidos", "EEUU"),
    ("South Korea", "Korea Republic", "Corea del Sur", "KOR"),
    ("North Korea", "Korea DPR", "DPR Korea", "DPRK", "Corea del Norte"),
    ("Iran", "IR Iran", "Irán"),
    ("Côte d'Ivoire", "Ivory Coast", "Cote d'Ivoire", "Cote dIvoire",
     "Costa de Marfil", "CIV"),
    ("Turkey", "Türkiye", "Turkiye", "Turquía", "TUR"),
    ("Czech Republic", "Czechia", "República Checa", "CZE"),
    ("China PR", "China", "CHN"),
    ("DR Congo", "Congo DR", "RD Congo"),
    ("Republic of Ireland", "Ireland", "Eire", "Irlanda", "IRL"),
    ("Bosnia and Herzegovina", "Bosnia-Herzegovina", "Bosnia", "BIH"),
    ("New Zealand", "Aotearoa New Zealand", "Nueva Zelanda", "NZL"),
    ("Cape Verde", "Cabo Verde"),
    ("Swaziland", "Eswatini"),
    ("Myanmar", "Burma", "Birmania"),
    ("Chinese Taipei", "Taiwan"),
    ("Hong Kong", "Hong Kong, China"),
    ("St Kitts and Nevis", "St. Kitts and Nevis", "Saint Kitts and Nevis"),
    ("St Lucia", "St. Lucia", "Saint Lucia", "Santa Lucía"),
    ("St Vincent and the Grenadines", "St. Vincent / Grenadines",
     "Saint Vincent and the Grenadines"),
    ("Trinidad and Tobago", "Trinidad & Tobago", "Trinidad y Tobago"),
    ("Antigua and Barbuda", "Antigua & Barbuda"),
    ("São Tomé and Príncipe", "Sao Tome and Principe"),
    ("Kyrgyz Republic", "Kyrgyzstan", "Kirguistán"),
    ("Brunei Darussalam", "Brunei"),
    ("North Macedonia", "Macedonia", "FYR Macedonia", "FYROM",
     "Macedonia del Norte", "MKD"),
    ("United Arab Emirates", "UAE", "Emiratos Árabes Unidos", "EAU"),
    ("Saudi Arabia", "KSA", "Arabia Saudita", "Arabia Saudí"),
    ("Dominican Republic", "Dom. Republic", "República Dominicana"),
    ("Central African Republic", "CAR", "República Centroafricana"),
    ("Papua New Guinea", "PNG", "Papúa Nueva Guinea"),
    ("Turks and Caicos Islands", "Turks and Caicos"),
    ("British Virgin Islands", "BVI"),
    ("US Virgin Islands", "USVI"),
    ("Algeria", "Argelia", "DZA"),
    ("Germany", "Alemania", "GER", "DEU", "Deutschland"),
    ("Japan", "Japón", "JPN"),
    ("Egypt", "Egipto", "EGY"),
    ("Morocco", "Marruecos", "MAR"),
    ("Switzerland", "Suiza", "SUI", "CHE"),
    ("Sweden", "Suecia", "SWE"),
    ("Denmark", "Dinamarca", "DEN", "DNK"),
    ("Norway", "Noruega", "NOR"),
    ("Finland", "Finlandia", "FIN"),
    ("Poland", "Polonia", "POL"),
    ("Greece", "Grecia", "GRE"),
    ("Netherlands", "Holanda", "Países Bajos", "NED", "NLD", "Holland"),
    ("Croatia", "Croacia", "CRO", "HRV"),
    ("Serbia", "SRB"),
    ("Scotland", "Escocia", "SCO"),
    ("Wales", "Gales", "WAL"),
    ("England", "Inglaterra", "ENG"),
    ("France", "Francia", "FRA"),
    ("Italy", "Italia", "ITA"),
    ("Spain", "España", "ESP"),
    ("Portugal", "POR", "PRT"),
    ("Belgium", "Bélgica", "BEL"),
    ("Argentina", "ARG"),
    ("Brazil", "Brasil", "BRA"),
    ("Colombia", "COL"),
    ("Uruguay", "URU", "URY"),
    ("Chile", "CHI", "CHL"),
    ("Paraguay", "PAR", "PRY"),
    ("Peru", "Perú", "PER"),
    ("Ecuador", "ECU"),
    ("Venezuela", "VEN"),
    ("Mexico", "México", "MEX"),
    ("Costa Rica", "CRC", "CRI"),
    ("Panama", "Panamá", "PAN"),
    ("Honduras", "HON", "HND"),
    ("Jamaica", "JAM"),
    ("Canada", "Canadá", "CAN"),
    ("Australia", "AUS"),
    ("South Africa", "Sudáfrica", "RSA", "ZAF"),
    ("Nigeria", "NGA"),
    ("Cameroon", "Camerún", "CMR"),
    ("Ghana", "GHA"),
    ("Senegal", "SEN"),
    ("Tunisia", "Túnez", "TUN"),
]

# Build bidirectional alias lookup: name.lower() → [all variants]
_ALIAS_LOOKUP = {}
for _group in _NAME_GROUPS:
    for _name in _group:
        _ALIAS_LOOKUP[_name.lower()] = list(_group)


def get_all_names(name):
    """Get all known name variants for a team."""
    key = name.strip().lower()
    if key in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[key]
    return [name.strip()]


def find_team_in_db(conn, user_name):
    """Find the canonical name for a team as it appears in the match database."""
    variants = get_all_names(user_name)
    for name in variants:
        cur = conn.execute(
            "SELECT DISTINCT home_team FROM matches WHERE home_team = ? LIMIT 1",
            (name,))
        row = cur.fetchone()
        if row:
            return row[0]
        cur = conn.execute(
            "SELECT DISTINCT away_team FROM matches WHERE away_team = ? LIMIT 1",
            (name,))
        row = cur.fetchone()
        if row:
            return row[0]
    return None


def suggest_teams(conn, query):
    """Suggest similar team names from the database."""
    q = query.lower()
    cur = conn.execute("SELECT DISTINCT home_team FROM matches")
    all_teams = sorted(set(row[0] for row in cur.fetchall()))
    suggestions = [t for t in all_teams if q in t.lower() or t.lower() in q]
    if not suggestions:
        words = q.split()
        suggestions = [t for t in all_teams
                       if any(w in t.lower() for w in words)]
    return suggestions[:10]
