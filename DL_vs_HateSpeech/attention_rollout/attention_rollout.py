from DL_vs_HateSpeech.attention_rollout.attention_plots import plot_attention_rollout

path = "DL_vs_HateSpeech/models/model_checkpoints/ModelV2_single_class_clip_16"

# Call plot attention rollout function
plot_attention_rollout(path, self_attn=True, blur=False, alpha_image=0.5, index=None, 
                       device="cpu", save_fig=True, show_fig=False)
print("Attention rollout complete.")