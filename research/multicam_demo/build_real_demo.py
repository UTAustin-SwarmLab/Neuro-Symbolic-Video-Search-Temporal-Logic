"""Inline the real-footage replay bundle into the tactical page template."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(args):
    data = (ROOT / "results" / "real" / args.data).read_text()
    html = (ROOT / args.template).read_text().replace("/*__DATA__*/null", data)
    out = ROOT / args.out
    out.write_text(html)
    print("wrote", out, "%.2f MB" % (out.stat().st_size / 1e6))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="demo_real.json")
    p.add_argument("--template", default="tactical_real_template.html")
    p.add_argument("--out", default="corridor_handoff_monitor.html")
    main(p.parse_args())
