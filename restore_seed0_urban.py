"""
Remet dans results/urban_results.csv les résultats de la graine 0 d'UrbanSound8K
(perdus quand le disque F: s'est déconnecté). Valeurs relevées dans les journaux
d'exécution du 15 et 16 octobre 2026 (arrondies à 0,1 point).
Les lignes déjà présentes ne sont pas dupliquées. Usage : python restore_seed0_urban.py
"""
from pathlib import Path
import crossgated as cg

SEED0 = {
    "svm":        [83.4, 84.3, 77.5, 82.5, 92.4, 80.2, 85.7, 82.4, 83.3, 88.5],
    "panns_only": [79.6, 80.4, 77.0, 77.3, 86.1, 72.2, 77.0, 84.2, 82.0, 85.5],
    "ast_only":   [85.3, 84.2, 75.7, 83.7, 86.0, 80.3, 86.4, 81.5, 85.2, 88.9],
    "concat":     [84.4, 84.6, 77.5, 80.3, 89.3, 81.0, 84.7, 84.9, 83.9, 88.2],
    "uncrossed":  [84.0, 87.2, 77.5, 81.3, 90.3, 81.5, 84.0, 84.7, 87.9, 89.7],
    "nofisher":   [83.6, 84.6, 79.7, 81.7, 92.5, 81.8, 87.1, 86.6, 87.6, 90.1],
    "crossgated": [85.2, 85.9, 80.5, 81.7, 91.2, 81.8, 86.2, 85.4, 89.0, 89.7],
}

csv_path = Path("results") / "urban_results.csv"
paths = [csv_path, cg.backup_of(csv_path)]
known = cg.read_rows(paths, cg.RESULT_COLS, cg.KEY_COLS)
if len(known):
    cg.sync_backup(known, paths[1])          # la sauvegarde sur C: reçoit d'abord tout ce qui existe déjà
done = {(r.model, int(r.seed), int(r.fold)) for r in known.itertuples()}
added = 0
for model, accs in SEED0.items():
    for fold, acc in enumerate(accs, start=1):
        if (model, 0, fold) not in done:
            cg.append_row(paths, {"model": model, "seed": 0, "fold": fold, "acc": round(acc / 100, 4)}, cg.RESULT_COLS)
            added += 1
print(f"{added} résultats de la graine 0 ajoutés (70 attendus si tout manquait).")
cg.summarize(csv_path)
