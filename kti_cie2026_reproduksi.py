"""
Reproduksi seluruh hasil KTI CIE 2026:
"Analisis Titik Impas Ekonomi Predictive Maintenance Berbasis Getaran pada Mesin
Berputar dengan Mempertimbangkan Akurasi Deteksi dan Alarm Fatigue"
Mencakup Bab IV (termasuk Subbab 4.4 ambang alarm optimal), Lampiran 5, 9A, dan 9B.
Cara menjalankan: python kti_cie2026_reproduksi.py
Kebutuhan: Python >= 3.9, numpy >= 1.22, scipy >= 1.8. Dijalankan penulis dengan
Python 3.13.18, numpy 2.5.3, scipy 1.18.1; waktu jalan sekitar 2 detik.
Di akhir keluaran, fungsi cek_angka_naskah() membandingkan hasil hitung dengan angka
yang dicetak di naskah dan berhenti dengan AssertionError bila ada yang tidak cocok.
Satuan biaya: juta rupiah per mesin per tahun. Unit analisis: satu mesin-tahun
(Δt = 1 tahun), paling banyak satu kegagalan dan satu alarm palsu per mesin-tahun.
"""
import math

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm

SEED = 2026


# -- parameter
def crf(i, n):
    """Faktor pemulihan modal (A/P, i, n)."""
    return i * (1 + i) ** n / ((1 + i) ** n - 1)


CAPEX, UMUR, BUNGA, OPEX = 20.0, 5, 0.10, 3.0
CS_DASAR = round(crf(BUNGA, UMUR) * CAPEX + OPEX, 2)  # 8,28 biaya tahunan ekuivalen
LAMBDA_IEEE, PORSI_BANTALAN = 0.0708, 0.44  # data sekunder (survei IEEE 1985)
MTBF = {
    "rendah": 1 / (LAMBDA_IEEE * PORSI_BANTALAN),  # 32,1 tahun (turunan)
    "menengah": 10.0,
    "tinggi": 4.0,
}  # asumsi skenario
DASAR = dict(Se=0.85, Sp=0.90, CF=100.0, CP=20.0, CI=2.0, CS=CS_DASAR, delta=0.5)
OPTIMISTIS = dict(Se=0.95, Sp=0.98, CF=200.0, CP=20.0, CI=0.5, CS=4.0, delta=0.0)
KONSERVATIF = dict(Se=0.70, Sp=0.80, CF=40.0, CP=20.0, CI=6.0, CS=16.0, delta=1.0)
RENTANG = dict(
    Se=(0.70, 0.95),
    Sp=(0.80, 0.98),
    CF=(40.0, 200.0),
    CP=(10.0, 30.0),
    CI=(0.5, 6.0),
    CS=(4.0, 16.0),
    delta=(0.0, 1.0),
)


# -- model
def p_eksponensial(mtbf, dt=1.0):  # Persamaan 1
    return 1 - math.exp(-dt / mtbf)


def ppv(p, Se, Sp):  # Persamaan 2
    tp, fp = p * Se, (1 - p) * (1 - Sp)
    return tp / (tp + fp)


def c_rtf(p, CF):  # Persamaan 3
    return p * CF


def c_pdm(p, Se, Sp, CF, CP, CI, CS):  # Persamaan 4 (delta = 0)
    return p * Se * CP + p * (1 - Se) * CF + (1 - p) * (1 - Sp) * CI + CS


def b_r(p, Se, Sp, CF, CP, CI, CS, delta=0.0):  # Persamaan 5 (delta=0) dan 9
    r = 1 - delta * (1 - ppv(p, Se, Sp))  # Persamaan 8
    return p * Se * r * (CF - CP) - (1 - p) * (1 - Sp) * CI - CS


def cf_star(p, Se, Sp, CP, CI, CS, delta=0.0):  # Persamaan 11 (delta=0) dan umum
    r = 1 - delta * (1 - ppv(p, Se, Sp))
    return CP + ((1 - p) * (1 - Sp) * CI + CS) / (p * Se * r)


def sp_min_d0(p, Se, CF, CP, CI, CS):  # Persamaan 6
    return 1 - (p * Se * (CF - CP) - CS) / ((1 - p) * CI)


def p_star_d0(Se, Sp, CF, CP, CI, CS):  # Persamaan 7
    return ((1 - Sp) * CI + CS) / (Se * (CF - CP) + (1 - Sp) * CI)


def p_star_kuadrat(Se, Sp, CF, CP, CI, CS, delta):  # Persamaan 10 (Lampiran 4)
    x, G = 1 - Sp, CF - CP
    d0, d1 = x, Se - x
    a2 = (1 - delta) * Se * G * d1 + delta * G * Se**2 + x * CI * d1
    a1 = (1 - delta) * Se * G * d0 - x * CI * (d1 - d0) - CS * d1
    a0 = -(x * CI + CS) * d0
    akar = (-a1 + math.sqrt(a1 * a1 - 4 * a2 * a0)) / (2 * a2)
    return akar, (a2, a1, a0)


def akar_numerik(f, a, b):
    return brentq(f, a, b, xtol=1e-12) if f(a) * f(b) < 0 else None


def weibull_p(t, eta, beta, dt=1.0):  # peluang gagal bersyarat
    return 1 - math.exp((t / eta) ** beta - ((t + dt) / eta) ** beta)


# -- ambang alarm optimal (Subbab 4.4)
def d_aksen(Se, Sp):  # d' = z(Se) - z(1 - Sp) = Phi^-1(Se) + Phi^-1(Sp)
    return norm.ppf(Se) + norm.ppf(Sp)


def titik_roc(t, d):  # Persamaan 13: kurva ROC binormal varians sama
    return norm.cdf(d - t), norm.cdf(t)  # (Se, Sp)


def b_ambang(t, p, b0, delta, d):
    se, sp = titik_roc(t, d)
    return b_r(p, **dict(b0, Se=se, Sp=sp), delta=delta)


def ambang_optimal(p, b0, delta, d):  # Persamaan 14, dicek dengan pencarian grid
    hasil = minimize_scalar(
        lambda t: -b_ambang(t, p, b0, delta, d), bounds=(-3, 6), method="bounded"
    )
    grid = np.linspace(-10, 12, 4401)
    t_grid = grid[np.argmax([b_ambang(t, p, b0, delta, d) for t in grid])]
    assert abs(hasil.x - t_grid) < 0.01, "optimum tidak global"
    return hasil.x


# -- uji sensitivitas struktural (Lampiran 9A dan 9B)
def b_gabungan(p, lam_dt, b0, delta, eps):  # Persamaan L9A.1, L9A.2, L9B.1
    r = 1 - delta * (1 - ppv(p, b0["Se"], b0["Sp"]))
    fp = (1 - p) * (1 - b0["Sp"]) * b0["CI"]
    return lam_dt * b0["Se"] * r * (eps * b0["CF"] - b0["CP"]) - fp - b0["CS"]


def eps_impas(p, lam_dt, b0, delta):
    r = 1 - delta * (1 - ppv(p, b0["Se"], b0["Sp"]))
    fp = (1 - p) * (1 - b0["Sp"]) * b0["CI"]
    return (b0["CP"] + (fp + b0["CS"]) / (lam_dt * b0["Se"] * r)) / b0["CF"]


def evppi(B, x, n_kelompok=50):  # EVPI parsial dengan pengelompokan kuantil
    batas = np.quantile(x, np.linspace(0, 1, n_kelompok + 1))
    idx = np.clip(np.searchsorted(batas, x, side="right") - 1, 0, n_kelompok - 1)
    n = np.bincount(idx, minlength=n_kelompok)
    rerata = np.bincount(idx, B, n_kelompok) / n
    return max((n / len(B) * np.maximum(rerata, 0)).sum() - max(B.mean(), 0.0), 0.0)


def fmt(x, nd=2):
    return f"{x:+.{nd}f}".replace(".", ",") if isinstance(x, float) else str(x)


# -- hasil
def main():
    b0 = {k: v for k, v in DASAR.items() if k != "delta"}
    P = {k: p_eksponensial(m) for k, m in MTBF.items()}
    print(f"C_S dasar = 20 x {crf(BUNGA, UMUR):.4f} + 3 = {CS_DASAR}")
    Se0, Sp0 = DASAR["Se"], DASAR["Sp"]

    print("\n[Tabel 3] per 1.000 mesin-tahun (Se=0,85; Sp=0,90)")
    for k, p in P.items():
        tp, fn = 1000 * p * Se0, 1000 * p * (1 - Se0)
        fp, tn = 1000 * (1 - p) * (1 - Sp0), 1000 * (1 - p) * Sp0
        print(
            f"  {k:9s} MTBF={MTBF[k]:5.1f} p={p:.4f} TP={tp:6.1f} "
            f"FN={fn:5.1f} FP={fp:5.1f} "
            f"TN={tn:6.1f} PPV={ppv(p, Se0, Sp0):.4f}"
        )

    print("\n[Tabel 4] skenario dasar")
    for k, p in P.items():
        cf_impas = b0["CP"] + ((1 - p) * (1 - b0["Sp"]) * b0["CI"] + b0["CS"]) / (
            p * b0["Se"]
        )
        spm = sp_min_d0(p, *(b0[x] for x in ("Se", "CF", "CP", "CI", "CS")))
        print(
            f"  {k:9s} C_RTF={c_rtf(p, b0['CF']):6.2f} C_PdM={c_pdm(p, **b0):6.2f} "
            f"B(d=0)={b_r(p, **b0):+6.2f} Br(d=0,5)={b_r(p, **b0, delta=0.5):+6.2f} "
            f"Br(d=1)={b_r(p, **b0, delta=1):+6.2f} CF*={cf_impas:6.1f} "
            f"Sp_min(d=0)={spm:.2f}"
        )

    print("\n[C_F* menurut delta] skenario dasar (ambang biaya kegagalan, juta Rp)")
    for k, p in P.items():
        nilai = [
            cf_star(p, *(b0[x] for x in ("Se", "Sp", "CP", "CI", "CS")), d)
            for d in (0.0, 0.5, 1.0)
        ]
        print(f"  {k:9s} " + "; ".join(f"{v:.1f}" for v in nilai))

    print("\n[Titik impas p*] skenario dasar")
    for dl in (0.0, 0.5, 1.0):
        pk, coef = p_star_kuadrat(**b0, delta=dl)
        pn = akar_numerik(lambda p: b_r(p, **b0, delta=dl), 1e-6, 0.999)
        print(
            f"  delta={dl}: kuadrat {coef[0]:.2f}p^2 {coef[1]:+.2f}p {coef[2]:+.3f} = 0"
            f" -> p*={pk:.4f} (numerik {pn:.4f}; "
            f"MTBF*={-1 / math.log(1 - pk):.2f} th; "
            f"substitusi B_r={b_r(pk, **b0, delta=dl):.1e})"
        )
    print(f"  Persamaan 7 (delta=0): {p_star_d0(**b0):.4f}")

    ph = P["tinggi"]
    print("\n[Sp minimum kelas tinggi]")
    for dl in (0.5, 1.0):
        s = akar_numerik(lambda sp: b_r(ph, **dict(b0, Sp=sp), delta=dl), 0.0, 1.0)
        print(f"  delta={dl}: Sp_min = {s:.4f}")

    print("\n[Kanal alarm palsu] kelas tinggi, Sp 0,98 -> 0,80, parameter lain = dasar")
    for dl in (0.0, 1.0):
        hi, lo = b_r(ph, **dict(b0, Sp=0.98), delta=dl), b_r(
            ph, **dict(b0, Sp=0.80), delta=dl
        )
        print(f"  delta={dl}: B_r {hi:.2f} -> {lo:.2f} (turun {hi - lo:.2f})")
    tak_langsung = (
        b_r(ph, **dict(b0, Sp=0.98), delta=1) - b_r(ph, **dict(b0, Sp=0.80), delta=1)
    ) - (1 - ph) * (0.98 - 0.80) * b0["CI"]
    langsung = (1 - ph) * (0.98 - 0.80) * b0["CI"]
    print(
        f"  biaya investigasi tambahan = {langsung:.2f}; "
        f"kanal tidak langsung (delta=1) = {tak_langsung:.2f}; "
        f"PPV {ppv(ph, Se0, 0.98):.3f} -> {ppv(ph, Se0, 0.80):.3f}"
    )

    print("\n[Tabel 5] skenario (delta: konservatif=1, dasar=0,5, optimistis=0)")
    for nm, par in (
        ("konservatif", KONSERVATIF),
        ("dasar", DASAR),
        ("optimistis", OPTIMISTIS),
    ):
        ps = akar_numerik(lambda p: b_r(p, **par), 1e-6, 0.999)
        print(
            f"  {nm:11s} "
            + " ".join(f"{k}={b_r(p, **par):+7.2f}" for k, p in P.items())
            + f" p*={'tidak ada' if ps is None else round(ps, 4)}"
        )

    print("\n[Tornado] kelas menengah, basis B_r (delta=0,5), satu parameter diubah")
    pm = P["menengah"]
    print(f"  basis = {b_r(pm, **DASAR):.2f}")
    baris = {"p": (b_r(P["rendah"], **DASAR), b_r(P["tinggi"], **DASAR))}
    for k, (lo, hi) in RENTANG.items():
        baris[k] = (
            b_r(pm, **dict(DASAR, **{k: lo})),
            b_r(pm, **dict(DASAR, **{k: hi})),
        )
    for k, (lo, hi) in sorted(baris.items(), key=lambda kv: -abs(kv[1][1] - kv[1][0])):
        print(
            f"  {k:6s} batas bawah={lo:+6.2f} batas atas={hi:+6.2f} "
            f"ayunan={abs(hi - lo):5.2f}"
        )

    print("\n[Weibull] rata-rata umur 10 tahun, ambang p* (delta=0)")
    ps0 = p_star_d0(**b0)
    for beta in (1.0, 2.0, 3.0):
        eta = 10 / math.gamma(1 + 1 / beta)
        umur = akar_numerik(lambda t: weibull_p(t, eta, beta) - ps0, 0.0, 60.0)
        print(
            f"  beta={beta}: eta={eta:.2f} th; umur saat p bersyarat = p*: "
            f"{'tidak tercapai' if umur is None else round(umur, 2)}"
        )

    print("\n[Kegagalan per 100 mesin-tahun] skenario dasar")
    for k, p in P.items():
        av = [100 * p * Se0 * (1 - dl * (1 - ppv(p, Se0, Sp0))) for dl in (0, 0.5, 1)]
        print(
            f"  {k:9s} RTF={100 * p:5.1f} "
            f"dicegah PdM (d=0;0,5;1)={av[0]:.1f};{av[1]:.1f};{av[2]:.1f} "
            f"alarm palsu={100 * (1 - p) * (1 - Sp0):.1f}"
        )

    # -- analisis probabilistik (100.000 sampel)
    LAYAK, EVPI, EVPPI = {}, {}, {}
    rng = np.random.default_rng(SEED)
    M = 100_000
    tri = lambda k: rng.triangular(
        RENTANG[k][0], DASAR[k], RENTANG[k][1], M
    )  # min, modus, maks
    Se, Sp, CF, CP, CI, CS = (tri(k) for k in ("Se", "Sp", "CF", "CP", "CI", "CS"))
    dl = rng.uniform(0.0, 1.0, M)  # delta ~ U(0,1)
    print(f"\n[Analisis probabilistik] N={M}, seed={SEED}, input saling bebas")
    for k, p in P.items():
        tp, fp = p * Se, (1 - p) * (1 - Sp)
        r = 1 - dl * (1 - tp / (tp + fp))
        B = p * Se * r * (CF - CP) - (1 - p) * (1 - Sp) * CI - CS
        n_layak = int((B > 0).sum())
        LAYAK[k] = n_layak
        X = dict(Se=Se, Sp=Sp, CF=CF, CP=CP, CI=CI, CS=CS, delta=dl)
        EVPI[k] = np.maximum(B, 0).mean() - max(B.mean(), 0.0)
        EVPPI[k] = {j: evppi(B, x) for j, x in X.items()}
        p5, med, p95 = np.percentile(B, [5, 50, 95])
        print(
            f"  {k:9s} median={med:+6.2f} P5={p5:+6.2f} P95={p95:+6.2f} "
            f"layak={n_layak}/{M} ({100 * n_layak / M:.2f}%)"
        )

    # -- verifikasi Monte Carlo (4 juta mesin-tahun)
    MAKS_Z = []
    rng = np.random.default_rng(SEED)
    N = 4_000_000
    print(f"\n[Verifikasi Monte Carlo] N={N} per kelas, seed={SEED}")
    for k, p in P.items():
        gagal = rng.random(N) < p
        alarm = np.where(
            gagal, rng.random(N) < DASAR["Se"], rng.random(N) < 1 - DASAR["Sp"]
        )
        for d in (0.0, 0.5, 1.0):
            ppv_emp = (gagal & alarm).sum() / alarm.sum()
            tindak = rng.random(N) < 1 - d * (1 - ppv_emp)
            biaya = np.full(N, b0["CS"])
            biaya += np.where(gagal & alarm & tindak, b0["CP"], 0.0)
            biaya += np.where(gagal & ~(alarm & tindak), b0["CF"], 0.0)
            biaya += np.where(~gagal & alarm, b0["CI"], 0.0)
            selisih = np.where(gagal, b0["CF"], 0.0) - biaya
            print(
                f"  {k:9s} delta={d}: analitik={b_r(p, **b0, delta=d):+.3f} "
                f"MC={selisih.mean():+.3f} "
                f"galat baku={selisih.std(ddof=1) / math.sqrt(N):.3f}"
            )
            MAKS_Z.append(
                abs(selisih.mean() - b_r(p, **b0, delta=d))
                / (selisih.std(ddof=1) / math.sqrt(N))
            )
    for k in P:
        teks = " ".join(f"{j}={v:.3f}" for j, v in EVPPI[k].items())
        print(f"  {k:9s} EVPI={EVPI[k]:.3f} EVPPI: {teks}")

    H = bagian_tambahan(P, b0)
    H["evpi"] = EVPI
    uji_properti()
    cek_angka_naskah(P, b0, LAYAK, max(MAKS_Z), H)


def bagian_tambahan(P, b0):
    """Subbab 4.2 (studi kasus), 4.4 (ambang optimal), Lampiran 9A dan 9B."""
    H = {}
    ph = P["tinggi"]
    d0 = d_aksen(b0["Se"], b0["Sp"])
    print(f"\n[Subbab 4.4] Ambang alarm optimal, ROC binormal, d'={d0:.3f}")
    for k, p in P.items():
        for dl in (0.0, 0.5, 1.0):
            t = ambang_optimal(p, b0, dl, d0)
            se, sp = titik_roc(t, d0)
            pv = ppv(p, se, sp)
            print(
                f"  {k:9s} delta={dl:.1f} t*={t:.3f} Se*={se:.3f} Sp*={sp:.3f} "
                f"PPV*={pv:.3f} r*={1 - dl * (1 - pv):.3f} "
                f"B*={b_ambang(t, p, b0, dl, d0):+.2f} "
                f"B(ambang dasar)={b_r(p, **b0, delta=dl):+.2f}"
            )
            H[(k, dl)] = (sp, b_ambang(t, p, b0, dl, d0))
    print(
        "  Kelas tinggi, ambang disetel untuk delta tertentu (baris) vs delta aktual:"
    )
    for dl in (0.0, 0.5, 1.0):
        t = ambang_optimal(ph, b0, dl, d0)
        nilai = [b_ambang(t, ph, b0, a, d0) for a in (0.0, 0.5, 1.0)]
        H[("setel", dl)] = nilai
        print(f"  disetel delta={dl:.1f}: " + " ".join(f"{v:+.2f}" for v in nilai))
    for d in (1.5, 3.0):
        t1 = ambang_optimal(ph, b0, 1.0, d)
        H[("d", d)] = b_ambang(t1, ph, b0, 1.0, d)
        print(f"  d'={d}: B maksimum kelas tinggi pada delta=1: {H[('d', d)]:+.2f}")

    lam = 3.0
    pp = 1 - math.exp(-lam)
    bp = [b_r(pp, **b0, delta=dl) for dl in (0.0, 0.5, 1.0)]
    H["pompa"] = (pp, ppv(pp, b0["Se"], b0["Sp"]), min(bp))
    print(
        f"\n[Subbab 4.2] Pompa 11-PA-7131A: p={pp:.3f} "
        f"PPV={H['pompa'][1]:.3f} B (delta 0; 0,5; 1)="
        + "; ".join(f"{v:+.2f}" for v in bp)
    )

    print("\n[Lampiran 9A dan 9B] Efektivitas tindakan eps dan kejadian berulang")
    for k, m in MTBF.items():
        p, lam_k = p_eksponensial(m), 1 / m
        for dl in (0.0, 0.5, 1.0):
            b_eps = [b_gabungan(p, p, b0, dl, e) for e in (0.70, 0.85, 1.00)]
            b_lam = [b_gabungan(p, lam_k, b0, dl, e) for e in (0.70, 0.85, 1.00)]
            H[("eps", k, dl)] = (eps_impas(p, p, b0, dl), b_eps, b_lam)
            print(
                f"  {k:9s} delta={dl:.1f} eps*={eps_impas(p, p, b0, dl):.3f} "
                f"eps*(lambda)={eps_impas(p, lam_k, b0, dl):.3f} "
                f"B(eps 0,70;0,85;1)="
                + ";".join(f"{v:+.2f}" for v in b_eps)
                + " gabungan="
                + ";".join(f"{v:+.2f}" for v in b_lam)
            )
    return H


def uji_properti():
    """Uji kasus batas pada fungsi model (verifikasi internal, bukan validasi)."""
    d = dict(Se=0.85, Sp=0.90, CF=100.0, CP=20.0, CI=2.0, CS=8.28)
    fp = 0.8 * 0.1 * 2.0
    uji = {
        "Se = 0 -> B = -FP*C_I - C_S": abs(b_r(0.2, **dict(d, Se=0.0)) + fp + 8.28),
        "C_F = C_P -> B = -FP*C_I - C_S": abs(
            b_r(0.2, **dict(d, CF=20.0), delta=0.5) + fp + 8.28
        ),
        "Sp = 1 -> r = 1 untuk semua delta": abs(
            b_r(0.2, **dict(d, Sp=1.0), delta=1) - b_r(0.2, **dict(d, Sp=1.0))
        ),
        "p = 0 -> B = -(1-Sp)C_I - C_S": abs(b_r(0.0, **d) + 0.2 + 8.28),
        "C_F* -> B_r = 0 (delta=0,5)": abs(
            b_r(
                0.1,
                **dict(d, CF=cf_star(0.1, 0.85, 0.90, 20.0, 2.0, 8.28, 0.5)),
                delta=0.5,
            )
        ),
    }
    for nama, galat in uji.items():
        assert galat < 1e-9, f"uji properti gagal: {nama}"
    assert b_r(0.2, **d) > b_r(0.2, **d, delta=0.5) > b_r(0.2, **d, delta=1)
    assert b_r(0.05, **dict(d, CI=0.0, CS=0.0), delta=1) > 0
    print(f"\n[Uji properti] {len(uji) + 2} uji kasus batas lulus")


def cek_angka_naskah(P, b0, layak, z_maks, H):
    """Membandingkan hasil hitung dengan angka yang tercetak di naskah."""
    ph, pm, pr = P["tinggi"], P["menengah"], P["rendah"]
    impas = {d: p_star_kuadrat(**b0, delta=d)[0] for d in (0.0, 0.5, 1.0)}
    spm = {
        d: akar_numerik(lambda sp: b_r(ph, **dict(b0, Sp=sp), delta=d), 0.0, 1.0)
        for d in (0.5, 1.0)
    }
    turun = b_r(ph, **dict(b0, Sp=0.98), delta=1) - b_r(
        ph, **dict(b0, Sp=0.80), delta=1
    )
    cfb = {k: b0[k] for k in ("Se", "Sp", "CP", "CI", "CS")}
    cek = [  # (keterangan, nilai hitung, nilai di naskah, toleransi pembulatan)
        ("C_S dasar", CS_DASAR, 8.28, 0.005),
        ("B rendah (d=0)", b_r(pr, **b0), -6.39, 0.005),
        ("B menengah (d=0)", b_r(pm, **b0), -1.99, 0.005),
        ("B tinggi (d=0)", b_r(ph, **b0), 6.61, 0.005),
        ("B_r tinggi (d=1)", b_r(ph, **b0, delta=1), 2.20, 0.005),
        ("p* d=0", impas[0.0], 0.124, 0.0005),
        ("p* d=0,5", impas[0.5], 0.155, 0.0005),
        ("p* d=1", impas[1.0], 0.187, 0.0005),
        ("Sp_min d=0,5", spm[0.5], 0.28, 0.005),
        ("Sp_min d=1", spm[1.0], 0.82, 0.005),
        ("penurunan Sp 0,98->0,80 (d=1)", turun, 5.94, 0.005),
        ("sampel layak tinggi", layak["tinggi"], 77662, 0),
        ("sampel layak menengah", layak["menengah"], 10967, 0),
        ("sampel layak rendah", layak["rendah"], 2, 0),
        ("C_F* menengah d=0,5", cf_star(pm, **cfb, delta=0.5), 162.1, 0.05),
        ("C_F* menengah d=1", cf_star(pm, **cfb, delta=1.0), 241.6, 0.05),
        ("d' dasar", d_aksen(b0["Se"], b0["Sp"]), 2.318, 0.0005),
        ("Sp* tinggi d=0", H[("tinggi", 0.0)][0], 0.544, 0.0005),
        ("Sp* tinggi d=0,5", H[("tinggi", 0.5)][0], 0.867, 0.0005),
        ("Sp* tinggi d=1", H[("tinggi", 1.0)][0], 0.936, 0.0005),
        ("B* tinggi d=0", H[("tinggi", 0.0)][1], 8.46, 0.005),
        ("B* tinggi d=0,5", H[("tinggi", 0.5)][1], 4.48, 0.005),
        ("B* tinggi d=1", H[("tinggi", 1.0)][1], 2.45, 0.005),
        ("ambang d=0, delta aktual 1", H[("setel", 0.0)][2], -2.35, 0.005),
        ("ambang d=0,5, delta aktual 0", H[("setel", 0.5)][0], 7.20, 0.005),
        ("ambang d=0,5, delta aktual 1", H[("setel", 0.5)][2], 1.77, 0.005),
        ("B maks d'=1,5, delta=1", H[("d", 1.5)], -1.76, 0.005),
        ("pompa p", H["pompa"][0], 0.950, 0.0005),
        ("pompa PPV", H["pompa"][1], 0.994, 0.0005),
        ("pompa B minimum", H["pompa"][2], 55.93, 0.005),
        ("eps* tinggi d=0", H[("eps", "tinggi", 0.0)][0], 0.649, 0.0005),
        ("eps* tinggi d=1", H[("eps", "tinggi", 1.0)][0], 0.835, 0.0005),
        ("B Poisson tinggi d=0", H[("eps", "tinggi", 0.0)][2][2], 8.56, 0.005),
        ("gabungan eps 0,70 d=0", H[("eps", "tinggi", 0.0)][2][0], 2.19, 0.005),
        ("gabungan eps 0,70 d=1", H[("eps", "tinggi", 1.0)][2][0], -0.92, 0.005),
        ("EVPI tinggi", H["evpi"]["tinggi"], 0.644, 0.0005),
        ("EVPI menengah", H["evpi"]["menengah"], 0.187, 0.0005),
    ]
    for nama, hitung, naskah, tol in cek:
        assert (
            abs(hitung - naskah) <= tol + 1e-12
        ), f"{nama}: hitung {hitung} != naskah {naskah}"
    assert z_maks < 3, f"selisih Monte Carlo terhadap analitik {z_maks:.1f} galat baku"
    print(
        f"\n[Pemeriksaan otomatis] {len(cek)} angka kunci cocok dengan naskah; "
        f"selisih Monte Carlo maksimum {z_maks:.1f} galat baku"
    )


if __name__ == "__main__":
    main()
