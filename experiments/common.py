import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def save(fig, name, data):
    fig.tight_layout()
    fig.savefig(RESULTS / f"{name}.png")
    fig.savefig(RESULTS / f"{name}.pdf")
    (RESULTS / f"{name}.json").write_text(json.dumps(data, indent=1))
    plt.close(fig)
