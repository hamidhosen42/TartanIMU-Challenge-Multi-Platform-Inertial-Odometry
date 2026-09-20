IMU Odometry Challenge — Technical Report Template
Version 2026-09-04   (this folder always holds the current version)

WHAT TO DOWNLOAD
  IMU_Challenge_Report_Template_2026-09-04.zip   the template — start here
  IMU_Challenge_Report_Template_2026-09-04.pdf   what it looks like, for reference

  zip  md5 e944cb8ec1aeab5633940c57a7f7a0e0
  pdf  md5 953d7b212c66d3aa1f01eb8685b33f3e

HOW TO USE IT
  Overleaf   New Project -> Upload Project -> select the .zip -> Recompile.
  Local      unzip, then run  pdflatex main.tex  three times.
  The zip is self-contained: main.tex + ieeeconf.cls, no .bib and no figure
  files are needed. Please do not download main.tex on its own — it will not
  compile without ieeeconf.cls.

THE FIVE THINGS PEOPLE GET WRONG
  1. Page limit. 6 pages excluding references, 7 pages in total including
     references. The appendix does not count towards either limit.
  2. Deadlines are NOT the same. Kaggle submissions and model weights close
     20 September 2026, 23:55 UTC. The report is due three days later,
     23 September 2026, 23:59 US Eastern Time (EDT, UTC-4).
  3. Where the numbers come from. Every number in Tables III, IV and V must be
     produced by the official scoring service, not by a local re-implementation:
     https://huggingface.co/spaces/Tartan-IMU/imu_odometry_challenge_scoring
     It returns ATE20, AVE and RTE for all 89 test sequences. Five submissions
     per team per day; your team name must match your Kaggle team name exactly,
     and teams with no Kaggle submission are not scored. State the MD5 of the
     submission CSV the numbers come from.
  4. The compliance declaration is required. Reports that leave it unanswered
     cannot be considered for the Top 10.
  5. Where to submit. The report is not uploaded on the website. Fill in the
     challenge Form as usual and attach the report at the end of the Form.

The Kaggle leaderboard is not the only criterion for the Top 10: the quality of
the report is also taken into account, and the Top 10 teams will be invited to
contribute to the forthcoming IMU Foundation Model white paper.

Latest rules and any change to the deadlines are posted in the Announcements
section of https://superodometry.com/imuchallenge
