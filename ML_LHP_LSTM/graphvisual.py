from graphviz import Digraph


def build_lstm_static_conditioning_graph():
    g = Digraph("LSTMStaticConditioning", format="png")
    g.attr(rankdir="TB", fontsize="12", dpi="150", nodesep="0.5", ranksep="0.7")

    # --- ノード定義 ---
    # 入力ノード
    g.node("static_x", "static_x\n(batch, static_dim)", shape="oval", style="filled", fillcolor="#fdf2e9")
    g.node("context_x", "context_x\n(batch, T_context, dynamic_dim)", shape="oval", style="filled", fillcolor="#e8f8f5")
    g.node("future_known", "future_known\n(batch, T_pred, dynamic_dim)", shape="oval", style="filled", fillcolor="#e8f8f5")

    # 静的 MLP -> init (h0, c0)
    with g.subgraph(name="cluster_static_mlp") as c:
        c.attr(label="Static MLP (init_hidden)", style="rounded", color="#f5cba7")
        c.node("static_mlp", "Linear → ReLU → Dropout\n→ Linear(2·L·H)", shape="box")
        c.node("h0c0", "(h0, c0)\n(num_layers, batch, hidden)", shape="box")

        c.edge("static_x", "static_mlp")
        c.edge("static_mlp", "h0c0")

    # 静的 embedding
    with g.subgraph(name="cluster_static_emb") as c:
        c.attr(label="Static Embedding", style="rounded", color="#aed6f1")
        c.node("static_emb", "Linear → ReLU → Dropout\n(static_emb_dim=64)", shape="box")
        c.node("static_emb_exp", "unsqueeze & expand\nover time steps", shape="box")

        c.edge("static_x", "static_emb")
        c.edge("static_emb", "static_emb_exp")

    # 共有 LSTM (単一インスタンス)
    g.node(
        "lstm_shared",
        "self.lstm (shared)\nnum_layers=2, hidden=256",
        shape="box",
        style="filled,bold",
        fillcolor="#d5dbdb"
    )

    # Context Encoder フェーズ
    with g.subgraph(name="cluster_encoder") as c:
        c.attr(label="Context Encoding Phase", style="rounded", color="#d2b4de")
        c.node(
            "context_cat",
            "concat([context_x, static_emb])\n(batch, T_context, dyn+emb)",
            shape="box"
        )
        c.node("lstm_enc", "LSTM forward pass", shape="box")
        c.node("hT", "(h_T, c_T)\nencoder final state", shape="box")

        c.edge("context_x", "context_cat")
        c.edge("static_emb_exp", "context_cat")
        c.edge("context_cat", "lstm_enc", label="context input")
        c.edge("h0c0", "lstm_enc", label="init state")
        c.edge("lstm_enc", "hT")

    # 共有 LSTM との関係 (圧縮表示)
    g.edge("lstm_shared", "lstm_enc", style="dashed", color="gray", label="weights")

    # Future Decoder フェーズ (autoregressive)
    with g.subgraph(name="cluster_decoder") as c:
        c.attr(label="Autoregressive Decoding Phase\n(for t = 1..T_pred)", style="rounded", color="#f9e79f")
        c.node(
            "future_cat",
            "concat([future_known[t], static_emb])\n(batch, 1, dyn+emb)",
            shape="box"
        )
        c.node(
            "lstm_step",
            "LSTM step (reuse self.lstm)",
            shape="box",
            style="dashed"
        )
        c.node(
            "out_head",
            "output_head\nLinear(H→H/2) → ReLU\n→ Dropout → Linear(H/2→4)",
            shape="box"
        )
        c.node("y_t", "y_pred[t]\n(batch, 4)", shape="box")

        c.edge("future_known", "future_cat")
        c.edge("static_emb_exp", "future_cat")
        c.edge("future_cat", "lstm_step")
        c.edge("lstm_step", "out_head")
        c.edge("out_head", "y_t")

    # Decoder が encoder state を引き継ぐ
    g.edge("hT", "lstm_step", label="init from encoder")

    # LSTM 共有を示す点線
    g.edge("lstm_shared", "lstm_step", style="dashed", color="gray", label="weights")

    # Autoregressive ループ (hidden state の更新)
    g.edge(
        "lstm_step",
        "lstm_step",
        label="h,c update (loop)",
        style="dashed",
        color="#e74c3c"
    )

    # 最終出力
    g.node("y_pred", "y_pred\n(batch, T_pred, 4)\n[GPP, NPP, NEP, Rh]", 
           shape="oval", style="filled", fillcolor="#d5f5e3")
    g.edge("y_t", "y_pred", label="stack over t")

    return g


if __name__ == "__main__":
    g = build_lstm_static_conditioning_graph()
    g.render("lstm_static_conditioning", cleanup=True)
    print("Saved to lstm_static_conditioning.png")
