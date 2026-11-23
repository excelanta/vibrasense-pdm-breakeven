"""
Perhitungan tambahan revisi akhir KTI CIE 2026 (Lampiran 6 bagian 9).
Memakai fungsi dari kti_cie2026_reproduksi.py tanpa mengubah angka lama.
  (A) Uji ketahanan ROC binormal varians tidak sama (Lampiran 9C, bagian c)
  (B) Uji kewajaran terhadap kasus terdokumentasi (Lampiran 12)
  (C) Ilustrasi dampak keberlanjutan: downtime tak terencana (Lampiran 13)
  (D) Cek silang harga sensor dengan rentang C_S (Lampiran 2)
  (E) Ilustrasi energi dan emisi dari jam henti yang dihindari (Lampiran 13)
  (F) Uji ketahanan terhadap rentang laju kegagalan survei lain (Lampiran 9C)
  (G) Aplikasi pada kasus pompa sentrifugal dengan biaya terdokumentasi (Lampiran 12)
Cara menjalankan: python kti_cie2026_tambahan.py (satu folder dengan berkas utama)
"""
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm

from kti_cie2026_reproduksi import (
    DASAR,
    MTBF,
    CS_DASAR,
    b_r,
    crf,
    ppv,
    p_eksponensial,
)

B0 = {k: v for k, v in DASAR.items() if k != "delta"}
P = {k: p_eksponensial(m) for k, m in MTBF.items()}
SE0, SP0 = B0["Se"], B0["Sp"]


# -- (A) ROC
def roc_tak_sama(t, b):
    """Sehat ~ N(0, 1), rusak ~ N(mu, b^2); mu dikalibrasi agar kurva
    melewati titik dasar (Se = 0,85; Sp = 0,90). b = 1 -> Persamaan 13."""
    mu = norm.ppf(SP0) + b * norm.ppf(SE0)
    return norm.cdf((mu - t) / b), norm.cdf(t), mu  # (Se, Sp, mu)


def optimum(p, delta, b):
    f = lambda t: -b_r(
        p,
        **dict(B0, Se=roc_tak_sama(t, b)[0], Sp=roc_tak_sama(t, b)[1]),
        delta=delta,
    )
    h = minimize_scalar(f, bounds=(-3, 6), method="bounded")
    grid = np.linspace(-10, 12, 4401)
    tg = grid[np.argmin([f(t) for t in grid])]
    assert abs(h.x - tg) < 0.01, "optimum tidak global"
    return h.x, -h.fun


def bagian_a():
    print("[A] Uji ROC varians tidak sama: kelas tinggi, delta = 0,5")
    p, dl = P["tinggi"], 0.5
    t_eq, _ = optimum(p, dl, 1.0)  # ambang optimal model varians sama
    for b in (0.8, 1.0, 1.25):
        se0, sp0, mu = roc_tak_sama(norm.ppf(SP0), b)
        assert abs(se0 - SE0) < 1e-12 and abs(sp0 - SP0) < 1e-12
        t, bmax = optimum(p, dl, b)
        se, sp, _ = roc_tak_sama(t, b)
        r = 1 - dl * (1 - ppv(p, se, sp))
        se_q, sp_q, _ = roc_tak_sama(t_eq, b)  # ambang varians sama dipakai
        b_q = b_r(p, **dict(B0, Se=se_q, Sp=sp_q), delta=dl)
        print(
            f"  b={b:4.2f} mu={mu:.3f} t*={t:.3f} Se*={se:.3f} Sp*={sp:.3f} "
            f"PPV*={ppv(p, se, sp):.3f} r*={r:.3f} Br*={bmax:+.2f} "
            f"| ambang varians sama: Sp={sp_q:.3f} Br={b_q:+.2f} "
            f"(rugi {bmax - b_q:.2f})"
        )
    print("  Pelengkap: Sp* (B_r*) kelas tinggi untuk delta = 0; 0,5; 1")
    for b in (0.8, 1.0, 1.25):
        isi = []
        for d in (0.0, 0.5, 1.0):
            t, bm = optimum(p, d, b)
            isi.append(f"{roc_tak_sama(t, b)[1]:.3f} ({bm:+.2f})")
        print(f"  b={b:4.2f}: " + "; ".join(isi))
    # ujung rentang pencarian Brent [-3; 6] untuk b = 0,8; 1; 1,25
    for b in (0.8, 1.0, 1.25):
        for t in (-3, 6):
            se, sp, _ = roc_tak_sama(t, b)
            print(f"  ujung t={t:+d}, b={b}: Se={se:.4f} Sp={sp:.4f}")


# -- (B) kewajaran
def bagian_b():
    print("\n[B] Kasus pompa 11-PA-7131A (Mulyadi et al., 2022)")
    lam = 3.0  # rusak tiga kali setahun
    p = 1 - np.exp(-lam)
    print(
        f"  lambda={lam} -> p={p:.4f}; PPV={ppv(p, SE0, SP0):.4f}; "
        f"MTBF={1 / lam:.3f} th"
    )
    for dl in (0.0, 0.5, 1.0):
        print(f"  delta={dl}: B_r={b_r(p, **B0, delta=dl):+.2f}")
    for cf in (40.0, 200.0):  # rentang C_F Tabel 2
        print(f"  C_F={cf:.0f}: B_r(delta=1)={b_r(p, **dict(B0, CF=cf), delta=1):+.2f}")


# -- (C) keberlanjutan
def bagian_c(jam_takrencana=24.0, jam_rencana=8.0):
    print(
        "\n[C] Ilustrasi downtime per 100 mesin-tahun "
        f"(asumsi ilustratif: {jam_takrencana:.0f} jam vs {jam_rencana:.0f} jam)"
    )
    for k in ("rendah", "tinggi"):
        p = P[k]
        for dl in (0.0, 1.0):
            r = 1 - dl * (1 - ppv(p, SE0, SP0))
            dicegah = 100 * p * SE0 * r
            jam = dicegah * (jam_takrencana - jam_rencana)
            print(
                f"  {k:6s} delta={dl}: kegagalan tak terencana {100 * p:.1f} -> "
                f"dicegah {dicegah:.1f}; downtime dihindari {jam:.0f} jam "
                f"({jam / 100:.2f} jam/mesin-tahun); alarm palsu "
                f"{100 * (1 - p) * (1 - SP0):.1f}"
            )
    for selisih in (8.0, 16.0, 40.0):  # uji rentang selisih durasi
        print(
            f"  selisih {selisih:.0f} jam: kelas tinggi delta=0 -> "
            f"{100 * P['tinggi'] * SE0 * selisih:.0f} jam per 100 mesin-tahun"
        )


# -- (D) harga sensor
def bagian_d(kurs=18067.6):
    print(f"\n[D] Harga daftar sensor (kurs Rp{kurs:,.1f}/US$)")
    for nama, usd in (
        ("Advantech WISE-2410-NB", 340.00),
        ("NCD wireless vibration-temperature V3", 385.95),
    ):
        rp = usd * kurs / 1e6
        I = 4 * rp + 8.0  # 4 sensor + komponen lain Lampiran 3
        cs = I * crf(0.10, 5) + 3.0
        br_tinggi = b_r(P["tinggi"], **dict(B0, CS=cs), delta=0.5)
        print(
            f"  {nama}: US${usd:.2f} = Rp{rp:.2f} juta/sensor; I={I:.1f}; "
            f"C_S={cs:.2f} (rentang Tabel 2: 4-16); "
            f"B_r tinggi delta=0,5 = {br_tinggi:+.2f}"
        )
    print(f"  pembanding asumsi dasar: Rp3,00 juta/sensor; C_S={CS_DASAR}")


# -- (E) energi dan emisi
def bagian_e(beban_idle_kw=10.0, fe_kg_per_kwh=0.87, selisih_jam=16.0):
    print(
        f"\n[E] Energi idle terbuang (asumsi ilustratif {beban_idle_kw:.0f} kW) dan emisi "
        f"(faktor emisi Jamali {fe_kg_per_kwh} kg CO2/kWh), per 100 mesin-tahun"
    )
    for k in ("rendah", "tinggi"):
        p = P[k]
        for dl in (0.0, 1.0):
            r = 1 - dl * (1 - ppv(p, SE0, SP0))
            jam = 100 * p * SE0 * r * selisih_jam
            kwh = jam * beban_idle_kw
            print(
                f"  {k:6s} delta={dl}: jam={jam:.1f}; energi={kwh:,.0f} kWh; "
                f"emisi={kwh * fe_kg_per_kwh / 1000:.2f} t CO2 "
                f"({kwh * fe_kg_per_kwh / 100:.1f} kg CO2/mesin-tahun)"
            )


# -- (F) rentang laju kegagalan survei
def bagian_f(laju_ieee=0.0708):
    print("\n[F] Semua MTBF kelas diskalakan dengan rasio laju survei lain / laju IEEE")
    for nama, laju in (
        ("batas bawah rentang 3 per 100", 0.030),
        ("EPRI 3,4 per 100", 0.034),
    ):
        f = laju / laju_ieee
        for k in ("rendah", "menengah", "tinggi"):
            m = MTBF[k] / f
            p = p_eksponensial(m)
            br = [b_r(p, **B0, delta=d) for d in (0.0, 0.5, 1.0)]
            print(
                f"  {nama} (faktor {f:.3f}) {k:8s}: MTBF={m:.1f} p={p:.3f} "
                f"B_r(delta=0; 0,5; 1)={br[0]:+.2f}; {br[1]:+.2f}; {br[2]:+.2f}"
            )
    print(
        "  MTBF impas kelas mana pun = -1/ln(1 - p*): 7,5 (delta=0) hingga 4,8 tahun (delta=1)"
    )


# -- (G) kasus biaya terdokumentasi
def bagian_g(mtbf_jam=521.4, jam_operasi=7300.0, cp=4.351, kurs=18067.6):
    print(
        "\n[G] Pompa sentrifugal (Ulum et al., 2023): MTBF bantalan 521,4 jam; "
        "biaya ganti bantalan Rp4,351 juta"
    )
    lam = jam_operasi / mtbf_jam
    p = 1 - np.exp(-lam)
    print(f"  lambda={lam:.1f} per tahun ({jam_operasi:.0f} jam operasi); p={p:.6f}")
    for d in (0.0, 0.5, 1.0):
        cf_star = brentq(lambda cf: b_r(p, **dict(B0, CF=cf, CP=cp), delta=d), cp, 1e4)
        print(f"  delta={d}: C_F impas = Rp{cf_star:.2f} juta per kegagalan tak terencana")
    print(
        f"  pembanding: perbaikan pompa kilang US$11.000 (Bloch) = "
        f"Rp{11000 * kurs / 1e6:.1f} juta"
    )


if __name__ == "__main__":
    bagian_a()
    bagian_b()
    bagian_c()
    bagian_d()
    bagian_e()
    bagian_f()
    bagian_g()
