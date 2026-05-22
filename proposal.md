  BC pretrain on top player replays
           ↓
  PPO self-play (recent self + checkpoints + static agents)
           ↓
  Submit to Kaggle → download match history
           ↓
  PPO continues, now with Kaggle opponents as implicit curriculum
           ↓
  If still stuck: add delta-planet shaping