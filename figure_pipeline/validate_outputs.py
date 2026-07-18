"""校验图清单、数量、文件配对与最低文件完整性。"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "thesis", "figures")
MANIFEST = os.path.join(FIGDIR, "figure_manifest.csv")


def main():
    m = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    assert len(m) == 19, f"图号应为19个，实际{len(m)}"
    assert m.figure.nunique() == 19, "图号存在重复"
    prog = m[m.type == "program"]
    hand = m[m.type == "hand_drawn"]
    assert len(prog) == 17, f"程序图应为17张，实际{len(prog)}"
    assert len(hand) == 2, f"手绘规格应为2张，实际{len(hand)}"

    expected = set()
    for _, r in prog.iterrows():
        stem = f"{r.figure}_{r.title}"
        for ext, min_bytes in [(".png", 50_000), (".pdf", 8_000)]:
            path = os.path.join(FIGDIR, stem + ext)
            assert os.path.isfile(path), f"缺少图文件：{path}"
            assert os.path.getsize(path) >= min_bytes, f"图文件异常偏小：{path}"
            expected.add(os.path.basename(path))

    actual = {n for n in os.listdir(FIGDIR)
              if n.startswith("图3-") and n.lower().endswith((".png", ".pdf"))}
    assert actual == expected, f"图目录存在缺失或旧编号文件：{sorted(actual ^ expected)}"

    for spec in hand.builder:
        path = os.path.join(ROOT, spec.replace("/", os.sep))
        assert os.path.isfile(path), f"缺少手绘规格：{path}"

    print("[ok] 图号19个：程序图17张、手绘规格2张")
    print("[ok] 17组 PNG/PDF 文件配对完整，无旧编号残留")


if __name__ == "__main__":
    main()
