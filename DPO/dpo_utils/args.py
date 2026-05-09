import argparse

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument

    g("--base_model_dir", required=True, help="Chemin du modèle base Qwen2.5-VL")
    g("--adapter_dir",    required=True, help="Chemin du checkpoint LoRA (adapter_model.safetensors)")
    g("--data_json",      required=True, help="JSONL unique avec tous les samples (sera split train/eval)")
    g("--val_ratio",      type=float, default=0.10, help="Fraction pour split eval/test (reproduit split)")
    g("--seed",           type=int,   default=42, help="Seed split + sampling")

    # (gardé pour compatibilité CLI, mais plus utilisé)
    g("--st_model", default="/home2020/home/beta/wlaemlin/hf_models/all_MiniLM-L6-v2",
      help="(INUTILISÉ) gardé pour compatibilité")

    g("--out_dir", required=True, help="Dossier de sortie (prefs + adapter DPO)")

    # Génération multi-candidats
    g("--num_candidates", type=int, default=4, help="Nombre de réponses générées par prompt")
    g("--temperature", type=float, default=0.8, help="Température sampling")
    g("--top_p", type=float, default=0.95, help="Top-p sampling")
    g("--max_new_tokens", type=int, default=512, help="Max tokens générés")

    # Préférences (options conservées mais non utilisées dans la nouvelle logique)
    g("--pref_strategy", choices=["best_vs_worst", "best_vs_random", "thresholded"], default="best_vs_worst",
      help="(INUTILISÉ) conservé pour compatibilité")
    g("--min_margin", type=float, default=0.05,
      help="(INUTILISÉ) conservé pour compatibilité")

    # DPO
    g("--beta", type=float, default=0.1, help="Paramètre beta DPO")
    g("--lr", type=float, default=2e-5, help="Learning rate")
    g("--epochs", type=int, default=1, help="Nombre d'époques DPO")
    g("--grad_accum_steps", type=int, default=1, help="Accumulation gradients")
    g("--max_train_samples", type=int, default=0, help="0 = tous, sinon limite sur nb de samples train")
    g("--length_norm", choices=["mean", "sum"], default="mean",
      help="Normalisation logprobs sur la longueur du completion")
    g("--save_every", type=int, default=200, help="Sauvegarde adapter toutes les N étapes (0 = jamais)")
    g("--fp16", action="store_true", help="Autocast fp16 (si CUDA)")

    # Modes
    g("--prefs_only", action="store_true", help="Génère seulement preferences.jsonl, sans DPO")
    g("--train_only", action="store_true", help="Suppose preferences.jsonl existe déjà, fait seulement DPO")
    g("--debug", action="store_true")

    # Préférence / objectif DPO
    g("--pref_model", choices=["bt", "pl", "lipo", "lipo_pl", "lipo_lambda"], default="pl",
      help="bt = Bradley-Terry pair-wise, pl = DPO-PL list-wise, lipo = LiPO full listwise PL")

    g("--lam", type=float, default=10.0,
                    help="DPOP penalty weight λ.")

    g( "--lam_non_top", type=float, default=0,
        help="Coefficient de pénalité DPOP appliqué aux sorties non-top.")

    # List-wise prefs (uniquement si pref_model=pl)
    g("--max_error_level", type=int, default=3,
      help="Niveau max k pour ordered_outputs: y0..yK (y_k = k citations fausses)")
    g("--max_offset", type=int, default=3,
      help="Décalage max timestamp pour temporal_confusion (list-wise)")

    return p

