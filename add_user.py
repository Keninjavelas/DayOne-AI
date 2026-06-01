import sys
sys.path.append('.')
from user_admin import load_app_config, save_app_config, create_user_record
from pathlib import Path

config = load_app_config(Path('config.yaml'))
try:
    create_user_record(config=config, username='testuser', password='password123', organization='org_acme', role='employee')
    save_app_config(Path('config.yaml'), config)
    print("Added testuser")
except Exception as e:
    print("Error:", e)
