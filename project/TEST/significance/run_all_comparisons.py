"""
Script Master pour exécuter TOUTES les comparaisons statistiques

Ce script lance tous les tests de significativité pour le projet HEARTS-FR.
Il exécute séquentiellement chaque comparaison et génère un rapport final.

Usage:
    python run_all_comparisons.py

Results:
    - Chaque comparaison génère ses propres fichiers dans results_significance/
    - Un rapport récapitulatif est créé à la fin
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Liste de toutes les comparaisons à exécuter
COMPARISONS = [
    {
        'script': 'compare_0_camembert_vs_albert.py',
        'name': 'CamemBERT-FR vs ALBERT-baseline',
        'category': 'Baseline Comparison'
    },
    {
        'script': 'compare_1_camembert_vs_distilbert.py',
        'name': 'CamemBERT-FR vs DistilBERT-multilingual',
        'category': 'Baseline Comparison'
    },
    {
        'script': 'compare_2_camembert_vs_logreg.py',
        'name': 'CamemBERT-FR vs Régression Logistique',
        'category': 'Baseline Comparison'
    },
    {
        'script': 'compare_3_camembert_vs_ablation_drop_unrelated.py',
        'name': 'CamemBERT-FR vs Ablation (sans unrelated)',
        'category': 'Ablation Study'
    },
    {
        'script': 'compare_4_camembert_vs_ablation_drop_neutral.py',
        'name': 'CamemBERT-FR vs Ablation (sans neutral)',
        'category': 'Ablation Study'
    },
    {
        'script': 'compare_6_camembert_maxlen.py',
        'name': 'CamemBERT max_length ablation',
        'category': 'Hyperparameter Study'
    },
    {
        'script': 'compare_7_camembert_learning_rate.py',
        'name': 'CamemBERT learning rate ablation',
        'category': 'Hyperparameter Study'
    },
    {
        'script': 'compare_8_camembert_epochs.py',
        'name': 'CamemBERT epochs ablation',
        'category': 'Hyperparameter Study'
    }
]

# ============================================================================
# FUNCTIONS
# ============================================================================

def run_comparison(script_path, comparison_name):
    """
    Exécute un script de comparaison et retourne le statut

    Args:
        script_path: Chemin vers le script Python
        comparison_name: Nom de la comparaison

    Returns:
        dict avec status ('success' ou 'failed') et message d'erreur si échec
    """
    print(f"\n{'=' * 80}")
    print(f"EXECUTION: {comparison_name}")
    print(f"Script: {script_path}")
    print(f"{'=' * 80}\n")

    start_time = time.time()

    try:
        # Exécuter le script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes max par comparaison
        )

        elapsed_time = time.time() - start_time

        # Afficher la sortie
        print(result.stdout)

        if result.returncode == 0:
            print(f"\n✓ SUCCÈS ({elapsed_time:.1f}s)")
            return {'status': 'success', 'time': elapsed_time, 'error': None}
        else:
            print(f"\n✗ ÉCHEC ({elapsed_time:.1f}s)")
            print("STDERR:")
            print(result.stderr)
            return {'status': 'failed', 'time': elapsed_time, 'error': result.stderr}

    except subprocess.TimeoutExpired:
        print("\n✗ ÉCHEC: Timeout (> 10 minutes)")
        return {'status': 'failed', 'time': 600, 'error': 'Timeout'}

    except Exception as e:
        print(f"\n✗ ÉCHEC: {str(e)}")
        return {'status': 'failed', 'time': 0, 'error': str(e)}


def generate_summary_report(results, output_file):
    """
    Génère un rapport récapitulatif de toutes les comparaisons

    Args:
        results: Liste des résultats de chaque comparaison
        output_file: Fichier de sortie pour le rapport
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RAPPORT RÉCAPITULATIF - TESTS DE SIGNIFICATIVITÉ STATISTIQUE\n")
        f.write("Projet HEARTS-FR\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Statistiques globales
        total = len(results)
        success = sum(1 for r in results if r['result']['status'] == 'success')
        failed = total - success
        total_time = sum(r['result']['time'] for r in results)

        f.write("STATISTIQUES GLOBALES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total de comparaisons: {total}\n")
        f.write(f"Succès: {success}\n")
        f.write(f"Échecs: {failed}\n")
        f.write(f"Temps total: {total_time:.1f}s ({total_time/60:.1f} minutes)\n\n")

        # Results by category
        categories = {}
        for r in results:
            cat = r['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)

        f.write("RÉSULTATS PAR CATÉGORIE\n")
        f.write("-" * 80 + "\n\n")

        for category, comps in categories.items():
            f.write(f"{category}:\n")
            for comp in comps:
                status_symbol = "✓" if comp['result']['status'] == 'success' else "✗"
                f.write(f"  {status_symbol} {comp['name']}\n")
                if comp['result']['status'] == 'failed':
                    f.write(f"     Erreur: {comp['result']['error'][:100]}...\n")
            f.write("\n")

        # Détails de chaque comparaison
        f.write("DÉTAILS DE CHAQUE COMPARISON\n")
        f.write("-" * 80 + "\n\n")

        for i, r in enumerate(results, 1):
            f.write(f"{i}. {r['name']}\n")
            f.write(f"   Script: {r['script']}\n")
            f.write(f"   Catégorie: {r['category']}\n")
            f.write(f"   Status: {r['result']['status']}\n")
            f.write(f"   Temps: {r['result']['time']:.1f}s\n")
            if r['result']['status'] == 'failed':
                f.write(f"   Erreur: {r['result']['error']}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("FIN DU RAPPORT\n")
        f.write("=" * 80 + "\n")


# ============================================================================
# MAIN PROGRAM
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("LAUNCHING ALL STATISTICAL COMPARISONS")
    print("Projet HEARTS-FR")
    print("=" * 80)

    # Répertoire courant (doit être le dossier significance/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print(f"\nRépertoire de travail: {script_dir}")
    print(f"Nombre de comparaisons à exécuter: {len(COMPARISONS)}\n")

    # Demander confirmation
    response = input("Voulez-vous continuer? (o/n): ")
    if response.lower() not in ['o', 'oui', 'y', 'yes']:
        print("Annulé par l'utilisateur.")
        sys.exit(0)

    # Exécuter toutes les comparaisons
    results = []
    start_time_total = time.time()

    for i, comp in enumerate(COMPARISONS, 1):
        script_path = os.path.join(script_dir, comp['script'])

        print(f"\n\n{'#' * 80}")
        print(f"COMPARISON {i}/{len(COMPARISONS)}")
        print(f"{'#' * 80}")

        # Vérifier que le script existe
        if not os.path.exists(script_path):
            print(f"✗ ERREUR: Script introuvable: {script_path}")
            result = {'status': 'failed', 'time': 0, 'error': 'Script not found'}
        else:
            result = run_comparison(script_path, comp['name'])

        results.append({
            'script': comp['script'],
            'name': comp['name'],
            'category': comp['category'],
            'result': result
        })

    # Temps total
    elapsed_total = time.time() - start_time_total

    # Generate the report récapitulatif
    report_file = "../../results_significance/summary_all_comparisons.txt"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    generate_summary_report(results, report_file)

    # Afficher le résumé final
    print("\n\n" + "=" * 80)
    print("RÉSUMÉ FINAL")
    print("=" * 80)

    total = len(results)
    success = sum(1 for r in results if r['result']['status'] == 'success')
    failed = total - success

    print(f"\nTotal de comparaisons: {total}")
    print(f"Succès: {success}")
    print(f"Échecs: {failed}")
    print(f"Temps total: {elapsed_total:.1f}s ({elapsed_total/60:.1f} minutes)")

    if failed > 0:
        print("\nFailed comparisons:")
        for r in results:
            if r['result']['status'] == 'failed':
                print(f"  ✗ {r['name']}")

    print(f"\nRapport complet sauvegardé: {report_file}")
    print("\n" + "=" * 80)
    print("TOUTES LES COMPARISONS TERMINÉES")
    print("=" * 80)
