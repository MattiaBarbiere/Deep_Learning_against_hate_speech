from DL_vs_HateSpeech.attention_rollout.attention_plots import plot_attention_rollout

# Selected model path
path = "DL_vs_HateSpeech/models/model_checkpoints/ModelV2_single_class_clip_16"

# Selected indexes
indexes = [813, 3, 4, 8, 10, 16, 17, 25, 27, 35, 38, 813]

# Loop through the selected indexes and process each meme
for i in indexes:
    print(f"Processing image {i}...")
    plot_attention_rollout(path, self_attn=True, blur=False, alpha_image=0.5, index=i, 
                        device="cpu", save_fig=False, show_fig=True)
    print("Attention rollout complete.")