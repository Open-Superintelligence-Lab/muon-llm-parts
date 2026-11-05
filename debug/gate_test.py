"""Does the gate see a known effect and stay quiet on noise? Uses open worlds and a temporary seal, so the real sealed worlds stay unread.

python3 debug/gate_test.py tiny    7 s code check, no assertions on scores
python3 debug/gate_test.py 5s      known effect: Muon's recipe beats Adam everywhere; noise: every lr changed by 0.1% is no win; a loser is no win; Muon through the sandbox scores the same as Muon
"""
import json, os, sys, tempfile
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE); T = os.environ.setdefault("SEAL_DIR", tempfile.mkdtemp())
import arms, gate



def main():
    TIER = sys.argv[1] if len(sys.argv) > 1 else "tiny"; WORLDS = [1, 2, 3, 4, 8]; R = os.path.join(T, "r.jsonl")
    ch = {r["arm"]: r["spec"] for r in gate.choices() if r["kind"] == "tune" and r["tier"] == "5s"}; mu, ad = ch["muon"], ch["all-adam"]; bad = ch["lion-all"]
    tiny = arms.parse(mu); [g.update(lr=g["lr"] * 1.001) for g in tiny.values()]; tiny = arms.show(tiny)
    box = arms.parse(mu); box["qk"]["rule"] = "x_muon_copy"; box = arms.show(box)
    left = gate.run([mu, ad, bad, tiny, box], TIER, WORLDS, R, 270); assert not left, f"{left} runs left"
    v = dict(known=gate.judge(ad, mu, TIER, WORLDS, R, 1), noise=gate.judge(mu, tiny, TIER, WORLDS, R, 1), loser=gate.judge(mu, bad, TIER, WORLDS, R, 1), box=gate.judge(mu, box, TIER, WORLDS, R, 1))
    for k, x in v.items(): print(k, json.dumps(x))
    if TIER != "tiny":
        k = v["known"]; assert k["mean"] > k["need"] and min(k["gaps"]) > 0, "the known effect was missed"
        assert not v["noise"]["win"] and not v["loser"]["win"] and not v["box"]["win"], "a false win"
        assert max(map(abs, v["box"]["gaps"])) < .001, "the sandbox changed Muon's score"
        print("the gate sees the known effect and stays quiet on the others")


if __name__ == "__main__":
    main()
