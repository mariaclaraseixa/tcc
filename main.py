"""
Ponto de entrada principal do TCC.
Oferece um menu interativo para executar as etapas do pipeline.
"""
import sys
import os

# Garante que a raiz do projeto esteja no path para imports de src.*
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CURRENT_EXPERIMENT


def _header():
    print("\n" + "=" * 60)
    print("  TCC — Detecção de Código Gerado por IA")
    print(f"  Experimento atual: {CURRENT_EXPERIMENT}")
    print("=" * 60)


def _menu():
    print("\n  [1] Rodar experimento (classificação)")
    print("  [2] Calcular métricas")
    print("  [3] Gerar tabela comparativa de modelos")
    print("  [4] Rodar análise de complexidade")
    print("  [5] Rodar tudo  (1 → 2 → 3 → 4)")
    print("  [0] Sair")
    return input("\n  Escolha uma opção: ").strip()


def _run_experiment():
    print("\n--- Iniciando experimento ---")
    from src.experiment import main as run
    run()


def _run_metrics():
    print("\n--- Calculando métricas ---")
    from src.metrics import main as run
    run()


def _run_compare():
    print("\n--- Gerando tabela comparativa ---")
    from src.compare import main as run
    run()


def _run_complexity():
    print("\n--- Análise de complexidade ---")
    from src.complexity import main as run
    run()


def main():
    _header()
    while True:
        choice = _menu()

        if choice == "1":
            _run_experiment()
        elif choice == "2":
            _run_metrics()
        elif choice == "3":
            _run_compare()
        elif choice == "4":
            _run_complexity()
        elif choice == "5":
            _run_experiment()
            _run_metrics()
            _run_compare()
            _run_complexity()
            print("\nPipeline completo finalizado.")
        elif choice == "0":
            print("\nSaindo. Até logo!\n")
            break
        else:
            print("\n  Opção inválida. Tente novamente.")

        input("\n  Pressione Enter para continuar...")
        _header()


if __name__ == "__main__":
    main()
