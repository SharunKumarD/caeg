import os
import pandas as pd
import urllib.request

def download_data():
    os.makedirs('data', exist_ok=True)
    
    # ==========================================
    # Dataset B: Modern PJM Hourly (2023-2024)
    # ==========================================
    print("Downloading Dataset: Modern PJM Hourly Energy Consumption...")
    try:
        pjm_url = "https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/refs/heads/main/datasets/pjm_in_zone.csv"
        df_pjm = pd.read_csv(pjm_url)
        
        # Determine column names (Nixtla typically uses unique_id, ds, y)
        zone_col = 'unique_id' if 'unique_id' in df_pjm.columns else ('zone' if 'zone' in df_pjm.columns else df_pjm.columns[0])
        time_col = 'ds' if 'ds' in df_pjm.columns else ('datetime' if 'datetime' in df_pjm.columns else df_pjm.columns[1])
        load_col = 'y' if 'y' in df_pjm.columns else ('load' if 'load' in df_pjm.columns else df_pjm.columns[2])
        
        # Filter for a single zone (e.g., AP-AP if it exists, else use the first unique zone)
        zones = df_pjm[zone_col].unique()
        target_zone = 'AP-AP' if 'AP-AP' in zones else zones[0]
        print(f"Filtering PJM data for zone: {target_zone}")
        
        df_pjm_filtered = df_pjm[df_pjm[zone_col] == target_zone].copy()
        
        # Rename and sort
        clean_pjm = pd.DataFrame({
            'datetime': pd.to_datetime(df_pjm_filtered[time_col]),
            'load': pd.to_numeric(df_pjm_filtered[load_col], errors='coerce')
        })
        clean_pjm = clean_pjm.sort_values('datetime').reset_index(drop=True)
        clean_pjm['load'] = clean_pjm['load'].interpolate(method='linear').bfill()
        
        # To match the fast-training subset size (~10000 rows)
        clean_pjm = clean_pjm.tail(10000).reset_index(drop=True)
        
        modern_path = 'data/dataset_modern.csv'
        clean_pjm.to_csv(modern_path, index=False)
        print(f"Dataset successfully saved to {modern_path}")
    except Exception as e:
        print(f"Error fetching Dataset: {e}")

if __name__ == "__main__":
    download_data()
