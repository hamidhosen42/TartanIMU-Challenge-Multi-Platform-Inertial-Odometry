"""Score a submission CSV on the organisers' official per-sequence scoring service and save the tables."""
import sys, json, shutil, time, pathlib, pandas as pd
from gradio_client import Client, handle_file
csv, team, tag = sys.argv[1], sys.argv[2], sys.argv[3]
out = pathlib.Path("official_scores"); out.mkdir(exist_ok=True)
for attempt in range(14):                                  # up to ~70 min (team-list sync)
    c = Client("Tartan-IMU/imu_odometry_challenge_scoring", verbose=False)
    overall, platform, seqs, csv_path, md = c.predict(handle_file(csv), team, api_name="/run")
    if overall and overall.get("data"):
        json.dump({"overall": overall, "platform": platform, "seqs": seqs, "markdown": md}, open(out / f"{tag}.json", "w"), indent=1)
        shutil.copy(csv_path, out / f"{tag}_per_sequence.csv")
        print(md); print(pd.DataFrame(overall["data"], columns=overall["headers"]).to_string(index=False))
        print(pd.DataFrame(platform["data"], columns=platform["headers"]).to_string(index=False)); break
    print(time.strftime("%H:%M"), md[:120].replace("\n", " ")); time.sleep(300)
