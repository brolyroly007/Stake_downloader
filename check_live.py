from curl_cffi import requests

# Streamers del usuario
streamers = [
    "elzeein", "sachauzumaki", "cristorata7", "kingteka", "milenkanolasco",
    "glogloking", "zullyy_cs", "benjaz", "neutroogg", "shuls_off",
    "diealis", "daarick", "noah_god", "luisormenoa27", "elsensei",
]

print("Verificando tus streamers en Kick...")
print("-" * 50)

live_count = 0
for s in streamers:
    try:
        r = requests.get(f'https://kick.com/api/v2/channels/{s}',
                        impersonate="chrome",
                        timeout=10)
        if r.status_code == 200:
            data = r.json()
            ls = data.get('livestream')
            if ls and ls.get('is_live'):
                print(f"🔴 LIVE: {s} - {ls.get('viewer_count', 0)} viewers - {ls.get('session_title', '')[:50]}")
                live_count += 1
            else:
                print(f"   offline: {s}")
        elif r.status_code == 404:
            print(f"   NO EXISTE: {s}")
        else:
            print(f"   error {r.status_code}: {s}")
    except Exception as e:
        print(f"   error: {s} - {e}")

print("-" * 50)
print(f"Total en vivo: {live_count}/{len(streamers)}")
