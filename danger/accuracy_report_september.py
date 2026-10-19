"""September 2026 accuracy report: our total (backend) vs GMR reported + fixed staff.

Data source:
- "Visitors Count UI" + "GMR Actual Data" are the store-verified numbers from the
  dashboard/owner (frozen snapshot provided by the user on 2026-09-30).
- "Our Total (Backend)" is computed live from the DB: distinct persons with a
  track session starting that IST day (customers + staff).

Usage:
  venv/bin/python danger/accuracy_report_september.py [--staff 40]

Percentages are computed here — never hand-typed.
"""

import argparse
import asyncio
import subprocess

# ── Store-verified snapshot (2026-09-30) ────────────────────────────────────
# (day, visitors_count_ui, gmr_actual_reported)
SNAPSHOT = [
    (1, 126, 108), (2, 120, 106), (3, 126, 154), (4, 116, 115),
    (5, 143, 110), (6, 122, 136), (7, 136, 161), (8, 137, 129),
    (9, 139, 116), (10, 141, 110), (11, 146, 115), (12, 140, 108),
    (13, 132, 119), (14, 156, 107), (15, 144, 121), (16, 145, 120),
    (17, 169, 130), (18, 159, 110), (19, 171, 116), (20, 171, 119),
    (21, 179, 125), (22, 170, 145), (23, 151, 138), (24, 181, 165),
    (25, 186, 141), (26, 172, 155), (27, 212, 149), (28, 190, 152),
    (29, 204, 107), (30, 242, 121),
]

QUERY = """
WITH ts AS (
  SELECT (started_at AT TIME ZONE 'Asia/Kolkata')::date AS day, person_identity_id
  FROM track_sessions
  WHERE started_at >= '2026-08-31 18:30:00+00' AND started_at < '2026-10-01 18:30:00+00'
    AND person_identity_id IS NOT NULL
)
SELECT EXTRACT(DAY FROM d)::int AS day,
  COUNT(DISTINCT ts.person_identity_id) FILTER (WHERE pi.is_staff = false) AS customers,
  COUNT(DISTINCT ts.person_identity_id) FILTER (WHERE pi.is_staff = true) AS staff,
  COUNT(DISTINCT ts.person_identity_id) AS total
FROM generate_series('2026-09-01'::date,'2026-09-30'::date,'1 day') d(days)
LEFT JOIN ts ON ts.day = d
LEFT JOIN person_identities pi ON pi.id = ts.person_identity_id
GROUP BY d ORDER BY d;
"""


def fetch_db_rows():
    out = subprocess.run(
        ["psql", "-h", "127.0.0.1", "-p", "5432", "-U", "retaileye_user",
         "-d", "retaileye_db", "-t", "-A", "-F", "|", "-c", QUERY],
        capture_output=True, text=True,
        env={"PGPASSWORD": "retaileye_pass", "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    if out.returncode != 0:
        raise SystemExit(f"psql failed: {out.stderr}")
    rows = {}
    for line in out.stdout.splitlines():
        parts = line.split("|")
        if len(parts) != 4:
            continue
        day = int(parts[0])
        rows[day] = {"customers": int(parts[1]), "staff": int(parts[2]), "total": int(parts[3])}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staff", type=int, default=40, help="fixed staff count added to GMR reported")
    args = ap.parse_args()

    db = fetch_db_rows()
    staff = args.staff

    print(f"\n## September 2026 — Our Total (Backend) vs GMR Actual + {staff} staff\n")
    print("| Date | Visitors Count UI | Our Total (Backend) | GMR Total (+%d) | Diff (±) | Error %% | Accuracy %% |" % staff)
    print("|---|---|---|---|---|---|---|")

    n = 0
    sum_ui = sum_our = sum_gmr = 0
    sum_err = 0.0
    sum_acc = 0.0
    neg = pos = 0
    for day, ui, reported in SNAPSHOT:
        our = db[day]["total"]
        gmr = reported + staff
        diff = our - gmr
        err = abs(diff) / gmr * 100.0
        acc = 100.0 - err
        sign = "+" if diff > 0 else ("−" if diff < 0 else "")
        dstr = f"{sign}{abs(diff)}" if diff else "0"
        print(f"| {day:02d}/09 | {ui} | {our} | {gmr} | {dstr} | {err:.1f}% | **{acc:.1f}%** |")
        n += 1
        sum_ui += ui; sum_our += our; sum_gmr += gmr
        sum_err += err; sum_acc += acc
        if diff > 0: pos += 1
        elif diff < 0: neg += 1

    print(f"| **Avg** | **{sum_ui/n:.1f}** | **{sum_our/n:.1f}** | **{sum_gmr/n:.1f}** | **{(sum_our-sum_gmr)/n:+.1f}** | **{sum_err/n:.1f}%** | **{sum_acc/n:.1f}%** |")
    net_err = abs(sum_our - sum_gmr) / sum_gmr * 100.0
    print(f"\nNet aggregate: our {sum_our} vs GMR {sum_gmr} → net {sum_our-sum_gmr:+d} ({net_err:.1f}% error, {100-net_err:.1f}% accuracy).")
    print(f"Days over-counted: {pos}, under-counted: {neg}, exact: {n-pos-neg}.\n")


if __name__ == "__main__":
    main()