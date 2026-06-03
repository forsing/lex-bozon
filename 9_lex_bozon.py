"""
9_lex_bozon.py

Finalni Qiskit/Aer kvantni (Higgs/bozon) model nad PRIRASTAJIMA lex niza.

Razlika od 8_quant_bozon_v2 (koji uci apsolutnu lex poziciju): 
ovde model uci raspodelu KORAKA dX = lex[t] - lex[t-1] (Brown/KW prirastaji),
pa je predikcija sledeci_lex = zadnji_lex + dX. 
To je 1D-niz-prirodni zadatak i daje drugaciju prognozu od combo-bozona.

Ulaz:
  /data/loto7lex_4626_k44.csv  (jedna kolona: lex)

Pristup:
  - dX = diff(lex); mapiranje dX u [0,1] pa u 25-bitni string
  - 25 qubita = 5 blokova x 5 qubita (bez sirenja na 35q)
  - Higgs/bozon kolo: peti blok kao globalno polje, Yukawa CRY+CRZ coupling,
    Goldstone CRZ prsten, symmetry breaking start
  - loss: multi-kernel MMD nad dX-distribucijom + kvantilni penalty + bit-MSE
  - exponential recency, conditional seed iz zadnjeg prirastaja (momentum)
  - sledeci_lex = zadnji_lex + dekodirani dX; filter placeholder + vec izvuceno
  - finalni izbor po lex-klaster score-u, derank u 7-kombinaciju

Output:
  9_lex_bozon.txt
  9_lex_bozon.png
"""

import csv
import math
import os
import random
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


T0 = time.time()
SEED = 39
CSV_PATH = "/Users/4c/Desktop/GHQ/data/loto7lex_4626_k44.csv"
HERE = os.path.dirname(os.path.abspath(__file__))
TXT_OUT = os.path.join(HERE, "9_lex_bozon.txt")
PNG_OUT = os.path.join(HERE, "9_lex_bozon.png")

N_NUMBERS = 39
K_PICK = 7
TOTAL_COMB = math.comb(N_NUMBERS, K_PICK)
PLACEHOLDER = (1, 2, 3, 4, 5, 6, 7)

N_QUBITS = 25
BLOCKS = 5
Q_PER_BLOCK = 5
HIGGS_BLOCK = 4
LAYERS = 4

TRAIN_ITERS = 100
TRAIN_SHOTS = 4096
FINAL_SHOTS = 150000
TOP_K = 12
TARGET_SAMPLE_N = 896
GEN_SAMPLE_N = 896
MMD_SIGMAS = (0.08, 0.18, 0.35)
HIGGS_VEV = 0.62
CLUSTER_WINDOW = 50000

# Prirastajni prostor: dX u [-C(39,7), +C(39,7)] -> [0,1] -> 25-bitni int.
STEP_SCALE = 2 ** N_QUBITS - 1
DX_SPAN = 2 * TOTAL_COMB


def fmt_time(seconds):
    return str(timedelta(seconds=int(round(seconds))))


def load_lex_csv(path):
    vals = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            cell = row[-1].strip()
            try:
                value = int(cell)
            except ValueError:
                skipped += 1
                continue
            if 1 <= value <= TOTAL_COMB:
                vals.append(value)
            else:
                skipped += 1
    if len(vals) < 200:
        raise ValueError("Premalo validnih lex vrednosti.")
    return np.array(vals, dtype=np.int64), skipped


def lex_derank(rank):
    r = int(rank) - 1
    combo = []
    start = 1
    for i in range(K_PICK):
        remaining = K_PICK - i - 1
        for x in range(start, N_NUMBERS + 1):
            cnt = math.comb(N_NUMBERS - x, remaining)
            if r < cnt:
                combo.append(x)
                start = x + 1
                break
            r -= cnt
    return tuple(combo)


def int_to_bitstring(value, n_bits=N_QUBITS):
    return format(int(value), f"0{n_bits}b")


def lex_region(lex_value):
    pct = 100.0 * int(lex_value) / TOTAL_COMB
    decile = min(10, max(1, int(math.ceil(pct / 10.0))))
    return f"D{decile} ({pct:.2f}%)"


def dx_to_unit(dx):
    """Prirastaj dX u [-TOTAL, +TOTAL] -> [0,1]."""
    return (float(dx) + TOTAL_COMB) / DX_SPAN


def unit_to_dx(u):
    """[0,1] -> prirastaj dX (ceo broj)."""
    return int(round(float(u) * DX_SPAN - TOTAL_COMB))


def recency_weights(n, tau=850.0):
    ages = np.arange(n - 1, -1, -1, dtype=np.float64)
    weights = np.exp(-ages / tau)
    weights /= weights.sum()
    return weights


def weighted_target_bits(unit_steps, weights):
    out = np.zeros(N_QUBITS, dtype=np.float64)
    for u, w in zip(unit_steps, weights):
        m = min(max(int(round(float(u) * STEP_SCALE)), 0), STEP_SCALE)
        bits = int_to_bitstring(m)
        out += w * np.fromiter((1.0 if b == "1" else 0.0 for b in bits), dtype=np.float64)
    return out


def weighted_target_sample(unit_steps, weights, sample_n=TARGET_SAMPLE_N):
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(unit_steps), size=sample_n, replace=True, p=weights)
    sample = unit_steps[pick].astype(np.float64)
    return sample.reshape(-1, 1)


def gaussian_mmd(x, y, sigma):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    xx = (x - x.T) ** 2
    yy = (y - y.T) ** 2
    xy = (x - y.T) ** 2
    denom = 2.0 * sigma * sigma
    kxx = np.exp(-xx / denom).mean()
    kyy = np.exp(-yy / denom).mean()
    kxy = np.exp(-xy / denom).mean()
    return float(kxx + kyy - 2.0 * kxy)


def multi_mmd(x, y):
    return float(np.mean([gaussian_mmd(x, y, s) for s in MMD_SIGMAS]))


def quantile_penalty(x, y):
    qs = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=np.float64)
    qx = np.quantile(x.reshape(-1), qs)
    qy = np.quantile(y.reshape(-1), qs)
    return float(np.mean((qx - qy) ** 2))


def counts_to_step_sample(counts, sample_n=GEN_SAMPLE_N):
    vals = []
    for bitstr, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
        clean = bitstr.replace(" ", "")
        if len(clean) != N_QUBITS:
            continue
        u = int(clean, 2) / STEP_SCALE
        vals.extend([u] * min(int(count), sample_n - len(vals)))
        if len(vals) >= sample_n:
            break
    if not vals:
        vals = [0.5]
    while len(vals) < sample_n:
        vals.append(vals[-1])
    return np.array(vals[:sample_n], dtype=np.float64).reshape(-1, 1)


def counts_to_bit_probs(counts):
    total = max(1, sum(counts.values()))
    probs = np.zeros(N_QUBITS, dtype=np.float64)
    for bitstr, count in counts.items():
        clean = bitstr.replace(" ", "")
        if len(clean) != N_QUBITS:
            continue
        probs += count * np.fromiter((1.0 if b == "1" else 0.0 for b in clean), dtype=np.float64)
    return probs / total


def params_per_layer():
    single = 2 * N_QUBITS
    intra_block_cry = BLOCKS * (Q_PER_BLOCK - 1)
    goldstone_ring = Q_PER_BLOCK
    higgs_mass_couplings = 2 * (BLOCKS - 1) * Q_PER_BLOCK
    matter_backreaction = 2 * (BLOCKS - 1)
    return single + intra_block_cry + goldstone_ring + higgs_mass_couplings + matter_backreaction


def build_bozon_qcbm(theta, seed_state):
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    seed_bits = int_to_bitstring(int(seed_state))
    for q, bit in enumerate(reversed(seed_bits)):
        if bit == "1":
            qc.x(q)

    p = 0
    higgs_start = HIGGS_BLOCK * Q_PER_BLOCK
    higgs_qubits = list(range(higgs_start, higgs_start + Q_PER_BLOCK))
    for layer in range(LAYERS):
        sign = 1.0 if layer % 2 == 0 else -1.0

        for q in range(N_QUBITS):
            field_shift = sign * HIGGS_VEV if q in higgs_qubits else 0.0
            qc.ry(float(theta[p] + field_shift), q)
            p += 1
            qc.rz(float(theta[p]), q)
            p += 1

        for block in range(BLOCKS):
            start = block * Q_PER_BLOCK
            for j in range(Q_PER_BLOCK - 1):
                qc.cry(float(theta[p]), start + j, start + j + 1)
                p += 1

        for j in range(Q_PER_BLOCK):
            qc.crz(float(theta[p]), higgs_qubits[j], higgs_qubits[(j + 1) % Q_PER_BLOCK])
            p += 1

        for block in range(BLOCKS - 1):
            start = block * Q_PER_BLOCK
            for j, hq in enumerate(higgs_qubits):
                qc.cry(float(theta[p]), hq, start + j)
                p += 1
                qc.crz(float(theta[p]), hq, start + j)
                p += 1

        h_center = higgs_start + 2
        for block in range(BLOCKS - 1):
            matter_center = block * Q_PER_BLOCK + 2
            qc.cry(float(theta[p]), matter_center, h_center)
            p += 1
            qc.crz(float(theta[p]), matter_center, h_center)
            p += 1

        for block in range(BLOCKS - 1):
            qc.cz(block * Q_PER_BLOCK + 2, h_center)

    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


def run_counts(theta, simulator, shots, seed_state, seed_offset=0):
    qc = build_bozon_qcbm(theta, seed_state)
    tqc = transpile(qc, simulator, optimization_level=1, seed_transpiler=SEED + seed_offset)
    result = simulator.run(tqc, shots=shots, seed_simulator=SEED + seed_offset).result()
    return result.get_counts()


def loss_parts(counts, target_sample, target_bits):
    generated_sample = counts_to_step_sample(counts)
    mmd = multi_mmd(generated_sample, target_sample)
    qpen = quantile_penalty(generated_sample, target_sample)
    bit_mse = float(np.mean((counts_to_bit_probs(counts) - target_bits) ** 2))
    total = float(mmd + 0.25 * qpen + 0.10 * bit_mse)
    return total, float(mmd), float(qpen), bit_mse


def init_theta_from_target(target_bits):
    rng = np.random.default_rng(SEED)
    ppl = params_per_layer()
    theta = np.zeros(LAYERS * ppl, dtype=np.float64)
    base_ry = 2.0 * np.arcsin(np.sqrt(np.clip(target_bits, 1e-6, 1.0 - 1e-6)))

    p = 0
    for layer in range(LAYERS):
        sign = 1.0 if layer % 2 == 0 else -1.0
        layer_scale = 1.0 / math.sqrt(layer + 1.0)
        for q in range(N_QUBITS):
            symmetry_break = sign * HIGGS_VEV if q // Q_PER_BLOCK == HIGGS_BLOCK else 0.0
            theta[p] = base_ry[q] * layer_scale + symmetry_break + rng.normal(0.0, 0.030)
            p += 1
            theta[p] = rng.normal(0.0, 0.09)
            p += 1
        for _ in range(BLOCKS * (Q_PER_BLOCK - 1)):
            theta[p] = rng.normal(0.0, 0.18)
            p += 1
        for _ in range(Q_PER_BLOCK):
            theta[p] = rng.normal(0.0, 0.20)
            p += 1
        for _ in range(2 * (BLOCKS - 1) * Q_PER_BLOCK):
            theta[p] = rng.normal(0.0, 0.28)
            p += 1
        for _ in range(2 * (BLOCKS - 1)):
            theta[p] = rng.normal(0.0, 0.20)
            p += 1
    return np.mod(theta, 2.0 * np.pi)


def spsa_train(theta0, target_sample, target_bits, simulator, seed_state):
    rng = np.random.default_rng(SEED)
    theta = theta0.copy()
    best_theta = theta.copy()
    best_loss = float("inf")
    losses, mmd_losses, q_losses, bit_losses = [], [], [], []

    for it in range(1, TRAIN_ITERS + 1):
        a = 0.11 / (it ** 0.36)
        c = 0.09 / (it ** 0.12)
        delta = rng.choice([-1.0, 1.0], size=theta.shape)

        counts_plus = run_counts(theta + c * delta, simulator, TRAIN_SHOTS, seed_state, 2 * it)
        counts_minus = run_counts(theta - c * delta, simulator, TRAIN_SHOTS, seed_state, 2 * it + 1)
        loss_plus, _, _, _ = loss_parts(counts_plus, target_sample, target_bits)
        loss_minus, _, _, _ = loss_parts(counts_minus, target_sample, target_bits)

        ghat = (loss_plus - loss_minus) / (2.0 * c) * delta
        theta = np.mod(theta - a * np.clip(ghat, -0.35, 0.35), 2.0 * np.pi)

        counts_eval = counts_plus if loss_plus <= loss_minus else counts_minus
        loss_now, mmd_now, q_now, bit_now = loss_parts(counts_eval, target_sample, target_bits)
        if loss_now < best_loss:
            best_loss = loss_now
            best_theta = theta.copy()

        losses.append(float(loss_now))
        mmd_losses.append(float(mmd_now))
        q_losses.append(float(q_now))
        bit_losses.append(float(bit_now))
        print(
            f"  SPSA iter {it:03d}/{TRAIN_ITERS}  "
            f"loss={loss_now:.8f}  mmd={mmd_now:.8f}  q={q_now:.8f}  bit={bit_now:.8f}"
        )

    return best_theta, losses, mmd_losses, q_losses, bit_losses


def cluster_support(valid_pairs, lex_val):
    lo = int(lex_val) - CLUSTER_WINDOW
    hi = int(lex_val) + CLUSTER_WINDOW
    return int(sum(count for value, count in valid_pairs if lo <= value <= hi))


def valid_sample_rows(counts, historical_set, last_lex):
    valid_pairs = []
    skipped_out = 0
    skipped_placeholder = 0
    skipped_seen = 0

    for bitstr, count in counts.items():
        clean = bitstr.replace(" ", "")
        if len(clean) != N_QUBITS:
            continue
        dx = unit_to_dx(int(clean, 2) / STEP_SCALE)
        lex_val = int(last_lex) + dx
        if not (1 <= lex_val <= TOTAL_COMB):
            skipped_out += int(count)
            continue
        combo = lex_derank(lex_val)
        if combo == PLACEHOLDER:
            skipped_placeholder += int(count)
            continue
        if lex_val in historical_set:
            skipped_seen += int(count)
            continue
        valid_pairs.append((int(lex_val), int(count)))

    rows = []
    seen_combos = set()
    for lex_val, count in valid_pairs:
        combo = lex_derank(lex_val)
        if combo in seen_combos:
            continue
        seen_combos.add(combo)
        support = cluster_support(valid_pairs, lex_val)
        score = (count / FINAL_SHOTS) * (1.0 + support / FINAL_SHOTS)
        rows.append(
            {
                "count": int(count),
                "cluster": int(support),
                "score": float(score),
                "prob": float(count) / FINAL_SHOTS,
                "lex": int(lex_val),
                "region": lex_region(int(lex_val)),
                "combo": combo,
            }
        )

    rows.sort(key=lambda row: (-float(row["score"]), -int(row["cluster"]), -int(row["count"])))
    return rows[:TOP_K], skipped_out, skipped_placeholder, skipped_seen


def make_png(losses, mmd_losses, q_losses, rows, target_bits, series):
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(1, len(losses) + 1), losses, marker="o", linewidth=1.2, label="loss")
    ax1.plot(range(1, len(mmd_losses) + 1), mmd_losses, linewidth=1.2, label="multi-MMD")
    ax1.plot(range(1, len(q_losses) + 1), q_losses, linewidth=1.0, label="quantile")
    ax1.set_title("9_lex_bozon SPSA loss")
    ax1.set_xlabel("iter")
    ax1.set_ylabel("loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(N_QUBITS), target_bits, color="#7c3aed")
    ax2.axvspan(HIGGS_BLOCK * Q_PER_BLOCK - 0.5, N_QUBITS - 0.5, color="#fde68a", alpha=0.25)
    ax2.set_title("Target bit amplitude + Higgs blok")
    ax2.set_xlabel("bit pozicija")
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", alpha=0.25)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    table_rows = [
        [
            i + 1,
            row["count"],
            row["cluster"],
            f"{row['score']:.8f}",
            row["lex"],
            row["region"],
            str(row["combo"]),
        ]
        for i, row in enumerate(rows)
    ]
    table = ax3.table(
        cellText=table_rows,
        colLabels=["rang", "count", "cluster", "score", "lex", "region", "kombinacija"],
        cellLoc="center",
        loc="center",
        colWidths=[0.05, 0.07, 0.08, 0.11, 0.13, 0.11, 0.36],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)
    for (r, _c), cell in table.get_celld().items():
        cell.set_edgecolor("#444444")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#312e81")
            cell.set_text_props(color="white", weight="bold")
        elif r == 1:
            cell.set_facecolor("#ede9fe")
            cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f3f4f6")

    fig.suptitle("9_lex_bozon - Qiskit QCBM Higgs/Yukawa nad 1D lex", fontweight="bold")
    fig.tight_layout()
    fig.savefig(PNG_OUT, dpi=200, bbox_inches="tight")
    plt.show()


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    print()
    print("=" * 72)
    print("9_lex_bozon - finalni Qiskit QCBM Higgs/Yukawa nad 1D lex nizom")
    print("=" * 72)
    print()

    lex_indices, skipped = load_lex_csv(CSV_PATH)
    historical_set = set(int(x) for x in lex_indices)
    increments = np.diff(lex_indices).astype(np.float64)
    unit_steps = (increments + TOTAL_COMB) / DX_SPAN
    weights = recency_weights(len(unit_steps))
    target_bits = weighted_target_bits(unit_steps, weights)
    target_sample = weighted_target_sample(unit_steps, weights)
    last_lex = int(lex_indices[-1])
    last_incr = float(increments[-1])
    seed_state = min(max(int(round(dx_to_unit(last_incr) * STEP_SCALE)), 0), STEP_SCALE)

    print(f"CSV:                 {CSV_PATH}")
    print(f"Validnih lex tacaka: {len(lex_indices)}")
    print(f"Preskoceno redova:   {skipped}")
    print(f"C(39,7):             {TOTAL_COMB:,}")
    print(f"Zadnji lex:          {last_lex:,}")
    print(f"Zadnji prirastaj dX: {last_incr:,.0f}")
    print(f"Qubita:              {N_QUBITS} = {BLOCKS} blokova x {Q_PER_BLOCK} qubita")
    print(f"Higgs blok:          blok {HIGGS_BLOCK + 1} / qubits {HIGGS_BLOCK * Q_PER_BLOCK}-{N_QUBITS - 1}")
    print(f"Layers:              {LAYERS}")
    print(f"Parametara:          {LAYERS * params_per_layer()}")
    print(f"Simulator:           AerSimulator qasm, shots train={TRAIN_SHOTS}, final={FINAL_SHOTS}")
    print()

    simulator = AerSimulator(method="automatic")
    theta0 = init_theta_from_target(target_bits)

    t_train = time.time()
    theta, losses, mmd_losses, q_losses, bit_losses = spsa_train(
        theta0, target_sample, target_bits, simulator, seed_state
    )
    train_seconds = time.time() - t_train

    print()
    print("Finalno semplovanje istreniranog 9_lex_bozon kola...")
    final_counts = run_counts(theta, simulator, FINAL_SHOTS, seed_state, 10_000)
    rows, skipped_out, skipped_placeholder, skipped_seen = valid_sample_rows(final_counts, historical_set, last_lex)

    if not rows:
        raise RuntimeError("Nema validnih novih sampled lex kandidata posle filtera.")

    main_row = rows[0]
    total_seconds = time.time() - T0

    lines = []
    lines.append("9_lex_bozon - finalni Qiskit QCBM Higgs/Yukawa nad 1D lex nizom")
    lines.append("=" * 72)
    lines.append("")
    lines.append("KORAK 1: lex niz (bijekcija kombinacija u 1..C(39,7))")
    lines.append("")
    lines.append(f"  CSV:                  {CSV_PATH}")
    lines.append(f"  Validnih lex tacaka:   {len(lex_indices)}")
    lines.append(f"  Preskoceno redova:     {skipped}")
    lines.append(f"  C(39,7):              {TOTAL_COMB:,}")
    lines.append(f"  Zadnji lex:            {last_lex:,}")
    lines.append(f"  Zadnji prirastaj dX:   {last_incr:,.0f}")
    lines.append("")
    lines.append("KORAK 2: Kvantni bozon model nad lex PRIRASTAJIMA (dX)")
    lines.append("")
    lines.append("  Model:                QCBM / parametrizovano kvantno kolo")
    lines.append("  Cilj:                 raspodela prirastaja dX = lex[t]-lex[t-1]")
    lines.append("  Predikcija:           sledeci_lex = zadnji_lex + dekodirani dX")
    lines.append("  Loss:                 multi-MMD + 0.25*quantile + 0.10*bit-MSE")
    lines.append("  Recency:              exponential weights nad svim prirastajima")
    lines.append(f"  Qubita:               {N_QUBITS} = {BLOCKS} blokova x {Q_PER_BLOCK}")
    lines.append(f"  Layers:               {LAYERS}")
    lines.append(f"  Parametara:           {len(theta)}")
    lines.append("  Conditional seed:     zadnji prirastaj dX enkodovan X-gateovima (momentum)")
    lines.append("  Higgs field:          peti 5q blok kao globalno polje")
    lines.append("  Symmetry breaking:    +/- Higgs VEV u inicijalizaciji i slojevima")
    lines.append("  Goldstone ring:       CRZ prsten unutar Higgs bloka")
    lines.append("  Yukawa coupling:      Higgs -> materija preko CRY+CRZ")
    lines.append("  Backreaction:         materija -> Higgs preko CRY+CRZ")
    lines.append("  Kandidati:            score po lex-klasteru, filter placeholder + vec izvuceno")
    lines.append(f"  SPSA iteracija:        {TRAIN_ITERS}")
    lines.append(f"  train shots:           {TRAIN_SHOTS}")
    lines.append(f"  final shots:           {FINAL_SHOTS}")
    lines.append(f"  initial loss:          {losses[0]:.8f}")
    lines.append(f"  final loss:            {losses[-1]:.8f}")
    lines.append(f"  best loss:             {min(losses):.8f}")
    lines.append(f"  final multi-MMD:       {mmd_losses[-1]:.8f}")
    lines.append(f"  final quantile loss:   {q_losses[-1]:.8f}")
    lines.append(f"  final bit MSE:         {bit_losses[-1]:.8f}")
    lines.append("")
    lines.append("Filter finalnih kandidata:")
    lines.append(f"  out-of-range shots:    {skipped_out}")
    lines.append(f"  placeholder shots:     {skipped_placeholder}")
    lines.append(f"  vec izvuceni shots:    {skipped_seen}")
    lines.append("")
    lines.append("PREDIKCIJA: NEXT / 9_lex_bozon")
    lines.append("=" * 72)
    lines.append("")
    lines.append("Glavna kvantna prognoza:")
    lines.append(f"  sampled count:         {main_row['count']}")
    lines.append(f"  cluster count:         {main_row['cluster']}")
    lines.append(f"  cluster score:         {main_row['score']:.8f}")
    lines.append(f"  sampled prob:          {main_row['prob']:.8f}")
    lines.append(f"  pred. lex:             {main_row['lex']:,}")
    lines.append(f"  lex-region:            {main_row['region']}")
    lines.append(f"  pred. kombinacija:     {main_row['combo']}")
    lines.append("  vec izvucena ranije:   NE (filtrirano)")
    lines.append("")
    lines.append("Top kvantni kandidati (cele kombinacije / lex-klasteri):")
    lines.append(
        f"  {'rang':<5}{'count':>8}{'cluster':>10}{'score':>13}"
        f"{'lex':>14}  {'region':<12} {'kombinacija':<30}"
    )
    for i, row in enumerate(rows, start=1):
        lines.append(
            f"  {i:<5}{row['count']:>8}{row['cluster']:>10}{row['score']:>13.8f}"
            f"{row['lex']:>14,}  {str(row['region']):<12} {str(row['combo']):<30}"
        )
    lines.append("")
    lines.append(f"Vreme treninga:       {fmt_time(train_seconds)} ({train_seconds:.1f} s)")
    lines.append(f"Ukupno vreme:         {fmt_time(total_seconds)} ({total_seconds:.1f} s)")
    lines.append(f"PNG:                  {PNG_OUT}")
    lines.append("")

    text = "\n".join(lines)
    print()
    print(text)
    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"TXT saved -> {TXT_OUT}")

    make_png(losses, mmd_losses, q_losses, rows, target_bits, lex_indices)
    print(f"PNG saved -> {PNG_OUT}")
    print()


if __name__ == "__main__":
    main()


"""
========================================================================
9_lex_bozon - finalni Qiskit QCBM Higgs/Yukawa nad 1D lex nizom
========================================================================

CSV:                 /data/loto7lex_4626_k44.csv
Validnih lex tacaka: 4626
Preskoceno redova:   1
C(39,7):             15,380,937
Zadnji lex:          2,770,100
Zadnji prirastaj dX: 673,249
Qubita:              25 = 5 blokova x 5 qubita
Higgs blok:          blok 5 / qubits 20-24
Layers:              4
Parametara:          492
Simulator:           AerSimulator qasm, shots train=4096, final=150000

  SPSA iter 001/100  loss=0.10400460  mmd=0.09350126  q=0.00695758  bit=0.08763943
  SPSA iter 002/100  loss=0.05277348  mmd=0.04124095  q=0.01200013  bit=0.08532497
  SPSA iter 003/100  loss=0.12259355  mmd=0.11327422  q=0.01167805  bit=0.06399824
  SPSA iter 004/100  loss=0.13109072  mmd=0.12234784  q=0.01148553  bit=0.05871490
  SPSA iter 005/100  loss=0.09508353  mmd=0.08544116  q=0.00661153  bit=0.07989492
  SPSA iter 006/100  loss=0.22657023  mmd=0.20839403  q=0.04894753  bit=0.05939311
  SPSA iter 007/100  loss=0.16825779  mmd=0.15827478  q=0.02016414  bit=0.04941972
  SPSA iter 008/100  loss=0.07708923  mmd=0.06928647  q=0.00915733  bit=0.05513426
  SPSA iter 009/100  loss=0.07379227  mmd=0.06508119  q=0.00662550  bit=0.07054708
  SPSA iter 010/100  loss=0.08064682  mmd=0.07130618  q=0.01043374  bit=0.06732197
  SPSA iter 011/100  loss=0.04928467  mmd=0.04206111  q=0.00401768  bit=0.06219139
  SPSA iter 012/100  loss=0.06068795  mmd=0.05272466  q=0.00493438  bit=0.06729696
  SPSA iter 013/100  loss=0.07394084  mmd=0.06368860  q=0.00866287  bit=0.08086522
  SPSA iter 014/100  loss=0.06843961  mmd=0.05851391  q=0.00723054  bit=0.08118062
  SPSA iter 015/100  loss=0.05292111  mmd=0.04249972  q=0.01130345  bit=0.07595530
  SPSA iter 016/100  loss=0.08886402  mmd=0.07920674  q=0.01155959  bit=0.06767384
  SPSA iter 017/100  loss=0.04744480  mmd=0.03621581  q=0.01034785  bit=0.08642030
  SPSA iter 018/100  loss=0.09944987  mmd=0.08611453  q=0.01964859  bit=0.08423199
  SPSA iter 019/100  loss=0.05642673  mmd=0.04559127  q=0.01224890  bit=0.07773232
  SPSA iter 020/100  loss=0.04808504  mmd=0.03780988  q=0.00899199  bit=0.08027165
  SPSA iter 021/100  loss=0.04288280  mmd=0.03303539  q=0.00721139  bit=0.08044567
  SPSA iter 022/100  loss=0.06751016  mmd=0.05864002  q=0.00620453  bit=0.07319003
  SPSA iter 023/100  loss=0.05456024  mmd=0.04424697  q=0.01122594  bit=0.07506780
  SPSA iter 024/100  loss=0.05410239  mmd=0.04469300  q=0.00836698  bit=0.07317643
  SPSA iter 025/100  loss=0.06142780  mmd=0.05392938  q=0.00503929  bit=0.06238600
  SPSA iter 026/100  loss=0.06747405  mmd=0.05916860  q=0.01008242  bit=0.05784850
  SPSA iter 027/100  loss=0.06212462  mmd=0.05397000  q=0.00889411  bit=0.05931099
  SPSA iter 028/100  loss=0.03585365  mmd=0.02747784  q=0.00686797  bit=0.06658817
  SPSA iter 029/100  loss=0.04944617  mmd=0.04175504  q=0.00795021  bit=0.05703582
  SPSA iter 030/100  loss=0.05707892  mmd=0.04951282  q=0.00826843  bit=0.05498993
  SPSA iter 031/100  loss=0.05091018  mmd=0.04271541  q=0.01108319  bit=0.05423976
  SPSA iter 032/100  loss=0.05342536  mmd=0.04473042  q=0.01134946  bit=0.05857582
  SPSA iter 033/100  loss=0.03411772  mmd=0.02675358  q=0.00786451  bit=0.05398018
  SPSA iter 034/100  loss=0.04780881  mmd=0.04026621  q=0.00972324  bit=0.05111787
  SPSA iter 035/100  loss=0.04254224  mmd=0.03541686  q=0.00673313  bit=0.05442096
  SPSA iter 036/100  loss=0.05018510  mmd=0.04335034  q=0.00701583  bit=0.05080805
  SPSA iter 037/100  loss=0.04126316  mmd=0.03352371  q=0.00686586  bit=0.06022980
  SPSA iter 038/100  loss=0.03601556  mmd=0.02821281  q=0.00757354  bit=0.05909364
  SPSA iter 039/100  loss=0.04116771  mmd=0.03349854  q=0.00795650  bit=0.05680043
  SPSA iter 040/100  loss=0.03674190  mmd=0.02947907  q=0.00704481  bit=0.05501627
  SPSA iter 041/100  loss=0.03947183  mmd=0.03168860  q=0.00658230  bit=0.06137655
  SPSA iter 042/100  loss=0.04027880  mmd=0.03186476  q=0.00793056  bit=0.06431400
  SPSA iter 043/100  loss=0.05116552  mmd=0.04311572  q=0.00777781  bit=0.06105342
  SPSA iter 044/100  loss=0.04612769  mmd=0.03917051  q=0.00618572  bit=0.05410750
  SPSA iter 045/100  loss=0.03513882  mmd=0.02754233  q=0.00758408  bit=0.05700478
  SPSA iter 046/100  loss=0.04658160  mmd=0.03835892  q=0.01024155  bit=0.05662295
  SPSA iter 047/100  loss=0.04446191  mmd=0.03679577  q=0.00758220  bit=0.05770592
  SPSA iter 048/100  loss=0.05080413  mmd=0.04218676  q=0.01131564  bit=0.05788457
  SPSA iter 049/100  loss=0.03874025  mmd=0.03203651  q=0.00594242  bit=0.05218128
  SPSA iter 050/100  loss=0.03538914  mmd=0.02816849  q=0.00658187  bit=0.05575182
  SPSA iter 051/100  loss=0.03419943  mmd=0.02803042  q=0.00561575  bit=0.04765069
  SPSA iter 052/100  loss=0.04096001  mmd=0.03217976  q=0.00631462  bit=0.07201597
  SPSA iter 053/100  loss=0.04040613  mmd=0.03148110  q=0.00779321  bit=0.06976723
  SPSA iter 054/100  loss=0.03621291  mmd=0.02763719  q=0.00745930  bit=0.06710899
  SPSA iter 055/100  loss=0.03276679  mmd=0.02567744  q=0.00627411  bit=0.05520828
  SPSA iter 056/100  loss=0.03865028  mmd=0.03119676  q=0.00720732  bit=0.05651694
  SPSA iter 057/100  loss=0.04141166  mmd=0.03358861  q=0.00639782  bit=0.06223597
  SPSA iter 058/100  loss=0.03999117  mmd=0.03226600  q=0.00696807  bit=0.05983155
  SPSA iter 059/100  loss=0.04697017  mmd=0.03896315  q=0.00871064  bit=0.05829358
  SPSA iter 060/100  loss=0.04148773  mmd=0.03438205  q=0.00686171  bit=0.05390247
  SPSA iter 061/100  loss=0.04466521  mmd=0.03730196  q=0.00711272  bit=0.05585078
  SPSA iter 062/100  loss=0.03817793  mmd=0.03073903  q=0.00726696  bit=0.05622164
  SPSA iter 063/100  loss=0.03775438  mmd=0.02996704  q=0.00655059  bit=0.06149688
  SPSA iter 064/100  loss=0.03053221  mmd=0.02314837  q=0.00634310  bit=0.05798065
  SPSA iter 065/100  loss=0.04078700  mmd=0.03213558  q=0.00902601  bit=0.06394914
  SPSA iter 066/100  loss=0.03810415  mmd=0.02991754  q=0.00780692  bit=0.06234883
  SPSA iter 067/100  loss=0.03912073  mmd=0.03154551  q=0.00699791  bit=0.05825745
  SPSA iter 068/100  loss=0.03733018  mmd=0.02976225  q=0.00641909  bit=0.05963161
  SPSA iter 069/100  loss=0.03355675  mmd=0.02611451  q=0.00713401  bit=0.05658733
  SPSA iter 070/100  loss=0.03527172  mmd=0.02702612  q=0.00742934  bit=0.06388260
  SPSA iter 071/100  loss=0.04457323  mmd=0.03574421  q=0.00817816  bit=0.06784474
  SPSA iter 072/100  loss=0.04630703  mmd=0.03851135  q=0.00785183  bit=0.05832727
  SPSA iter 073/100  loss=0.03532186  mmd=0.02855828  q=0.00598334  bit=0.05267747
  SPSA iter 074/100  loss=0.03501854  mmd=0.02731350  q=0.00762076  bit=0.05799845
  SPSA iter 075/100  loss=0.03248990  mmd=0.02549551  q=0.00556052  bit=0.05604259
  SPSA iter 076/100  loss=0.04424164  mmd=0.03638072  q=0.00731958  bit=0.06031026
  SPSA iter 077/100  loss=0.04511446  mmd=0.03879525  q=0.00678424  bit=0.04623158
  SPSA iter 078/100  loss=0.04173127  mmd=0.03423910  q=0.00684627  bit=0.05780605
  SPSA iter 079/100  loss=0.02931842  mmd=0.02240708  q=0.00418975  bit=0.05863905
  SPSA iter 080/100  loss=0.05920718  mmd=0.05148362  q=0.00705351  bit=0.05960179
  SPSA iter 081/100  loss=0.06090398  mmd=0.05442217  q=0.00636108  bit=0.04891543
  SPSA iter 082/100  loss=0.06677969  mmd=0.05832510  q=0.00736170  bit=0.06614166
  SPSA iter 083/100  loss=0.05495143  mmd=0.04765530  q=0.00523761  bit=0.05986733
  SPSA iter 084/100  loss=0.04165434  mmd=0.03518728  q=0.00517055  bit=0.05174427
  SPSA iter 085/100  loss=0.04361572  mmd=0.03636989  q=0.00480251  bit=0.06045201
  SPSA iter 086/100  loss=0.04247675  mmd=0.03501540  q=0.00518750  bit=0.06164473
  SPSA iter 087/100  loss=0.05017679  mmd=0.04355902  q=0.00560245  bit=0.05217152
  SPSA iter 088/100  loss=0.04024365  mmd=0.03248118  q=0.00744235  bit=0.05901881
  SPSA iter 089/100  loss=0.03917286  mmd=0.03174184  q=0.00571953  bit=0.06001136
  SPSA iter 090/100  loss=0.04039393  mmd=0.03385658  q=0.00488244  bit=0.05316741
  SPSA iter 091/100  loss=0.02481954  mmd=0.01870404  q=0.00358091  bit=0.05220275
  SPSA iter 092/100  loss=0.03065297  mmd=0.02408448  q=0.00429546  bit=0.05494625
  SPSA iter 093/100  loss=0.03582507  mmd=0.02943123  q=0.00585232  bit=0.04930756
  SPSA iter 094/100  loss=0.03016964  mmd=0.02312055  q=0.00630350  bit=0.05473212
  SPSA iter 095/100  loss=0.03729939  mmd=0.03073886  q=0.00673042  bit=0.04877922
  SPSA iter 096/100  loss=0.04760510  mmd=0.03998277  q=0.00826908  bit=0.05555059
  SPSA iter 097/100  loss=0.03354840  mmd=0.02656566  q=0.00503059  bit=0.05725086
  SPSA iter 098/100  loss=0.04331444  mmd=0.03706752  q=0.00547873  bit=0.04877241
  SPSA iter 099/100  loss=0.04795692  mmd=0.04099103  q=0.00537259  bit=0.05622734
  SPSA iter 100/100  loss=0.03594263  mmd=0.02886705  q=0.00700217  bit=0.05325035

Finalno semplovanje istreniranog 9_lex_bozon kola...

9_lex_bozon - finalni Qiskit QCBM Higgs/Yukawa nad 1D lex nizom
========================================================================

KORAK 1: lex niz (bijekcija kombinacija u 1..C(39,7))

  CSV:                   /data/loto7lex_4626_k44.csv
  Validnih lex tacaka:   4626
  Preskoceno redova:     1
  C(39,7):               15,380,937
  Zadnji lex:            2,770,100
  Zadnji prirastaj dX:   673,249

KORAK 2: Kvantni bozon model nad lex PRIRASTAJIMA (dX)

  Model:                QCBM / parametrizovano kvantno kolo
  Cilj:                 raspodela prirastaja dX = lex[t]-lex[t-1]
  Predikcija:           sledeci_lex = zadnji_lex + dekodirani dX
  Loss:                 multi-MMD + 0.25*quantile + 0.10*bit-MSE
  Recency:              exponential weights nad svim prirastajima
  Qubita:               25 = 5 blokova x 5
  Layers:               4
  Parametara:           492
  Conditional seed:     zadnji prirastaj dX enkodovan X-gateovima (momentum)
  Higgs field:          peti 5q blok kao globalno polje
  Symmetry breaking:    +/- Higgs VEV u inicijalizaciji i slojevima
  Goldstone ring:       CRZ prsten unutar Higgs bloka
  Yukawa coupling:      Higgs -> materija preko CRY+CRZ
  Backreaction:         materija -> Higgs preko CRY+CRZ
  Kandidati:            score po lex-klasteru, filter placeholder + vec izvuceno
  SPSA iteracija:        100
  train shots:           4096
  final shots:           150000
  initial loss:          0.10400460
  final loss:            0.03594263
  best loss:             0.02481954
  final multi-MMD:       0.02886705
  final quantile loss:   0.00700217
  final bit MSE:         0.05325035

Filter finalnih kandidata:
  out-of-range shots:    59980
  placeholder shots:     0
  vec izvuceni shots:    28

PREDIKCIJA: NEXT / 9_lex_bozon
========================================================================

Glavna kvantna prognoza:
  sampled count:         28
  cluster count:         3671
  cluster score:         0.00019124
  sampled prob:          0.00018667
  pred. lex:             2,938,097
  lex-region:            D2 (19.10%)
  pred. kombinacija:     (2, 3, 8, 9, 15, 17, 34)
  vec izvucena ranije:   NE (filtrirano)

Top kvantni kandidati (cele kombinacije / lex-klasteri):
  rang    count   cluster        score           lex  region       kombinacija                   
  1          28      3671   0.00019124     2,938,097  D2 (19.10%)  (2, 3, 8, 9, 15, 17, 34)      
  2          25      3671   0.00017075     2,938,089  D2 (19.10%)  (2, 3, 8, 9, 15, 17, 26)      
  3          25      2325   0.00016925     3,403,723  D3 (22.13%)  (2, 4, 14, 19, 23, 29, 37)    
  4          23      5176   0.00015862     3,869,364  D3 (25.16%)  (2, 6, 11, 16, 18, 27, 37)    
  5          23      3671   0.00015709     2,938,104  D2 (19.10%)  (2, 3, 8, 9, 15, 18, 20)      
  6          22      5395   0.00015194     3,884,378  D3 (25.25%)  (2, 6, 12, 14, 29, 33, 35)    
  7          22      3552   0.00015014     2,922,607  D2 (19.00%)  (2, 3, 7, 14, 20, 27, 34)     
  8          22      2398   0.00014901     3,418,744  D3 (22.23%)  (2, 4, 15, 25, 31, 38, 39)    
  9          21      2895   0.00014270     4,860,707  D4 (31.60%)  (2, 14, 15, 19, 26, 32, 37)   
  10         21      1935   0.00014181     5,101,034  D4 (33.16%)  (3, 4, 5, 9, 11, 23, 33)      
  11         20      3555   0.00013649     2,923,069  D2 (19.00%)  (2, 3, 7, 14, 23, 33, 36)     
  12         20      3345   0.00013631     4,350,011  D3 (28.28%)  (2, 9, 10, 11, 14, 22, 26)    

Vreme treninga:       0:23:33 (1412.9 s)
Ukupno vreme:         0:25:25 (1524.7 s)
PNG:                  /9_lex_bozon.png

TXT saved -> /9_lex_bozon.txt
PNG saved -> /9_lex_bozon.png
"""




"""
Analiza 9_lex_bozon:

Ovo je dX-bozon: 
ulaz za model su prirastaji dX = lex[t] - lex[t-1], a izlaz se vraća kao zadnji_lex + dX.

Trening
Loss je pao 0.1040 → 0.0359, best 0.0248. To je solidno.
multi-MMD = 0.0289 znači da je model naučio deo raspodele prirastaja, ali nije jako oštar.
bit MSE = 0.0533 je pristojan i bolji od mezona; bitovi su bolje usklađeni nego kod 9_lex_mezon.

Filter
out-of-range = 59,980 od 150k, oko 40%. 
To nije malo, ali je prihvatljivo za dX model jer deo generisanih koraka izbaci lex van opsega.
vec izvuceni = 28, zanemarljivo. Glavna kombinacija nije ranije izvučena.
Predikcija Glavna prognoza:

lex = 2,938,097
kombinacija = (2, 3, 8, 9, 15, 17, 34)
region D2 (19.10%)

Top kandidati jasno prave prvi klaster oko 2.938M:

2,938,097
2,938,089
2,938,104
To je vrlo usko jezgro, razlika samo nekoliko lex pozicija. 
Drugi klaster ide oko 3.86M–3.88M, a treći oko 3.40M–3.42M.

Zaključak: 
Daje niži region, na granici D2/D3, sa najjačim fokusom oko 2.938M. 

Glavna kombinacija za ovaj model je:
(2, 3, 8, 9, 15, 17, 34)
"""



"""
9_lex_bozon — dX (prirastajni) lex model, SEED=39.

Uči raspodelu prirastaja dX = lex[t] − lex[t-1] (Brown/KW koraci),
a NE apsolutnu lex poziciju. Time se razlikuje od 8_quant_bozon_v2.

  - Cilj: dX mapiran u [0,1] -> 25-bitni string; weighted_target_bits /
    weighted_target_sample rade nad unit_steps (prirastajima), ne nad lex.
  - Generisani uzorak (counts_to_step_sample): izmereni bitstring kao korak
    u [0,1], bez odsecanja opsega — to je raspodela koju loss poredi.
  - Conditional seed: enkodira zadnji prirastaj dX (momentum), ne zadnji lex.
  - Dekodiranje (valid_sample_rows): izmereni bitstring -> dX ->
    kandidat_lex = zadnji_lex + dX (umesto kandidat_lex = izmereni indeks).
  - Predikcija: sledeći_lex = zadnji_lex + dekodirani dX.
  - SEED = 39 (kao i svuda); razlika u rezultatu dolazi iz modela, ne iz seed-a.
"""
