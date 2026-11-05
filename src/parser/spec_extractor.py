import re
from typing import Dict, Any, Tuple, Optional

CANONICAL_BRANDS = {
    'CORSAIR': 'Corsair',
    'KINGSTON': 'Kingston',
    'G.SKILL': 'G.Skill',
    'TEAMGROUP': 'TeamGroup',
    'KLEVV': 'KLEVV',
    'CRUCIAL': 'Crucial',
    'LEXAR': 'Lexar',
    'HIKSEMI': 'Hiksemi',
    'THERMALTAKE': 'Thermaltake',
    'APACER': 'Apacer',
    'ADATA': 'ADATA',
    'SILICON POWER': 'Silicon Power',
    'PATRIOT': 'Patriot',
    'ZADAK': 'Zadak',
    'GEIL': 'GeIL',
    'ACER': 'Acer',
    'COLORFUL': 'Colorful',
    'TRANSCEND': 'Transcend',
    'PNY': 'PNY',
    'SAMSUNG': 'Samsung'
}

def extract_brand(title: str) -> str:
    upper = title.upper()
    for b_key, b_name in CANONICAL_BRANDS.items():
        if re.search(r'\b' + re.escape(b_key) + r'\b', upper):
            return b_name
    
    # Specific brand aliases
    if 'G-SKILL' in upper or 'GSKILL' in upper or 'RIPJAWS' in upper or 'TRIDENT' in upper:
        return 'G.Skill'
    if 'TEAM GROUP' in upper or 'T-FORCE' in upper or 'T-CREATE' in upper or 'DELTA RGB' in upper:
        return 'TeamGroup'
    if 'FURY' in upper or 'BEAST' in upper or 'RENEGADE' in upper:
        return 'Kingston'
    if 'VENGEANCE' in upper or 'DOMINATOR' in upper:
        return 'Corsair'
    if 'LANCER' in upper or 'XPG' in upper:
        return 'ADATA'
    
    return 'Other'

def extract_generation(title: str) -> str:
    upper = title.upper()
    if 'DDR5' in upper:
        return 'DDR5'
    if 'DDR4' in upper:
        return 'DDR4'
    if 'DDR3' in upper:
        return 'DDR3'
    return 'Unknown'

def extract_capacity_and_modules(title: str) -> Tuple[Optional[int], int]:
    # Match patterns like: (8GBx2), (16GBx1), 32GB(16GBX2), (2x8GB)
    mod_match1 = re.search(r'\(?\s*(\d+)\s*GB\s*[xX*]\s*(\d+)\s*\)?', title, re.I)
    if mod_match1:
        per_mod = int(mod_match1.group(1))
        num_mods = int(mod_match1.group(2))
        return per_mod * num_mods, num_mods
    
    mod_match2 = re.search(r'\(?\s*(\d+)\s*[xX*]\s*(\d+)\s*GB\s*\)?', title, re.I)
    if mod_match2:
        num_mods = int(mod_match2.group(1))
        per_mod = int(mod_match2.group(2))
        return per_mod * num_mods, num_mods

    # Match simple total capacity like "32GB" or "16 GB"
    cap_match = re.search(r'\b(\d+)\s*GB\b', title, re.I)
    if cap_match:
        cap = int(cap_match.group(1))
        is_kit = any(k in title.upper() for k in ['KIT', 'DUAL', '2X', '8X2', '16X2', '32X2'])
        modules = 2 if (is_kit and cap in [16, 32, 64, 96, 128]) else 1
        return cap, modules

    return None, 1

def extract_speed(title: str, gen: str) -> int:
    # Common speeds in MHz or MT/s
    spd_match = re.search(
        r'\b(2133|2400|2666|2933|3000|3200|3600|4000|4400|4800|5200|5600|6000|6400|6600|6800|7200|7600|8000)\s*(?:MHz|MT/s|Mbps)?\b',
        title,
        re.I
    )
    if spd_match:
        return int(spd_match.group(1))

    # Standard fallback according to DDR gen
    if gen == 'DDR5':
        return 5600
    elif gen == 'DDR4':
        return 3200
    return 2666

def extract_cas_latency(title: str, gen: str, speed: int) -> int:
    cl_match = re.search(r'\b(?:CL|C)\s*[-:]?\s*(\d{2})\b', title, re.I)
    if cl_match:
        cl = int(cl_match.group(1))
        if 14 <= cl <= 46:
            return cl

    # Realistic standard latency fallback
    if gen == 'DDR5':
        if speed >= 6400:
            return 32
        elif speed >= 6000:
            return 36
        elif speed >= 5600:
            return 38
        else:
            return 40
    else:  # DDR4
        if speed >= 3600:
            return 18
        elif speed >= 3200:
            return 16
        else:
            return 19

def extract_rgb(title: str) -> int:
    return 1 if re.search(r'\bRGB\b', title, re.I) else 0

def parse_ram_product(raw_item: Dict[str, Any]) -> Dict[str, Any]:
    title = raw_item.get('title', '')
    price = raw_item.get('price', 0)
    url = raw_item.get('url', '')
    cat = raw_item.get('category', 'Desktop RAM')

    gen = extract_generation(title)
    brand = extract_brand(title)
    capacity, modules = extract_capacity_and_modules(title)
    speed = extract_speed(title, gen)
    cas = extract_cas_latency(title, gen, speed)
    rgb = extract_rgb(title)

    return {
        'name': title,
        'brand': brand,
        'generation': gen,
        'capacity_gb': capacity,
        'modules': modules,
        'speed_mhz': speed,
        'cas_latency': cas,
        'has_rgb': rgb,
        'form_factor': 'SO-DIMM' if 'NOTEBOOK' in cat.upper() or 'SO-DIMM' in title.upper() else 'UDIMM',
        'price_thb': price,
        'url': url
    }
