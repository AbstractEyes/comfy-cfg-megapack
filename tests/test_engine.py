"""CPU tests for cfg_megapack/engine.py and papers.py (no ComfyUI, no GPU).

Run from the repository folder:  python tests/test_engine.py
The GPU is hidden before torch is imported and the file asserts CUDA is unavailable."""
import os
import subprocess
import sys
import tempfile
import json
import math
import types

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch  # noqa: E402

assert torch.cuda.is_available() is False, "tests must run on CPU only"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cfg_megapack import engine  # noqa: E402
from cfg_megapack import cfg_variants as cv  # noqa: E402

torch.manual_seed(0)
B, C, H, W = 2, 4, 16, 16
TOL = 2e-4


def close(a, b, tol=TOL):
    return float((a.float() - b.float()).abs().max()) <= tol * (1 + float(b.float().abs().max()))


class FakeSampling:
    """Stands in for ComfyUI's EPS model_sampling (percent -> sigma on a log-linear grid)."""
    sigma_max, sigma_min = 14.6, 0.03

    def percent_to_sigma(self, p):
        if p <= 0:
            return 999999999.9
        if p >= 1:
            return 0.0
        return math.exp(math.log(self.sigma_max) * (1 - p) + math.log(self.sigma_min) * p)


class FakeModel:
    model_sampling = FakeSampling()


def batch(sigma=2.0):
    x0_c = torch.randn(B, C, H, W)
    x0_u = x0_c * 0.7 + 0.3 * torch.randn(B, C, H, W)
    x = x0_c + sigma * torch.randn(B, C, H, W)
    return x, x0_c, x0_u, torch.full((B,), sigma)


def run(plan, x, c, u, sig, w=6.0, sigmas=None, **kw):
    rt = engine.GuidanceRuntime(plan)
    mo = {"transformer_options": {"sample_sigmas": sigmas if sigmas is not None else torch.tensor([14.6, 6.0, 2.0, 0.5, 0.0])}}
    return rt.guided_x0(x, c, u, sig, w, model=FakeModel(), model_options=mo, **kw)


def plan_with(**stages):
    p = engine.empty_plan()
    p.update(stages)
    return p


def cfg(c, u, w):
    return u + w * (c - u)


# ---------------------------------------------------------------------------

def test_space_roundtrips():
    x, c, _, sig = batch(1.7)
    sv = sig.view(B, 1, 1, 1)
    for flow in (False, True):
        s = sv.clamp(max=0.95) if flow else sv
        for space in engine.SPACES:
            d = engine.to_space(c, x, s, space, flow)
            back = engine.from_space(d, x, s, space, flow)
            assert close(back, c, 1e-4), (space, flow)


def test_linear_cfg_is_space_invariant():
    x, c, u, sig = batch(3.0)
    sv = sig.view(B, 1, 1, 1)
    ref = cfg(c, u, 5.0)
    for space in engine.SPACES:
        dc, du = engine.to_space(c, x, sv, space, False), engine.to_space(u, x, sv, space, False)
        out = engine.from_space(cfg(dc, du, 5.0), x, sv, space, False)
        assert close(out, ref, 1e-4), space


def test_empty_plan_is_plain_cfg():
    x, c, u, sig = batch()
    assert close(run(engine.empty_plan(), x, c, u, sig), cfg(c, u, 6.0))


def test_mix_rules_neutral_settings_reduce_to_cfg():
    x, c, u, sig = batch()
    ref = cfg(c, u, 6.0)
    neutral = {
        "standard": {},
        "apg": {"eta": 1.0, "norm_threshold": 0.0, "momentum": 0.0},
        "power_law": {"alpha": 0.0},
        "magnitude_damped": {"alpha": 0.0},
    }
    for rule, knobs in neutral.items():
        for space in ("auto", "x0", "eps", "v"):
            p = plan_with(mix={"kind": "rule", "rule": rule, "scale": -1.0, "knobs": knobs, "space": space})
            assert close(run(p, x, c, u, sig), ref, 5e-4), (rule, space)


def test_mix_rules_match_library():
    x, c, u, sig = batch()
    sv = sig.view(B, 1, 1, 1)
    cases = [("apg", "apg", {"eta": 0.0, "norm_threshold": 15.0, "momentum": 0.0}, "x0"),
             ("tangential_damping", "tcfg", {}, "eps"),
             ("angle_limit", "adg", {"max_angle": 0.7}, "x0"),
             ("mahiro", "mahiro", {}, "x0"),
             ("cfg_zero_star", "cfg_zero_star", {"zero_init_steps": 0}, "eps")]
    for rule, reg, knobs, space in cases:
        p = plan_with(mix={"kind": "rule", "rule": rule, "scale": -1.0, "knobs": knobs, "space": "auto"})
        out = run(p, x, c, u, sig)
        dc, du = engine.to_space(c, x, sv, space, False), engine.to_space(u, x, sv, space, False)
        lib = engine.from_space(cv.apply_variant(reg, dc, du, w=6.0, ctx={"x_t": x, "sigma": sig, "step": 1,
                                                                          "space": "x0" if space == "x0" else "noise"},
                                                 **knobs).float(), x, sv, space, False)
        assert close(out, lib, 5e-4), rule


def test_mix_scale_override():
    x, c, u, sig = batch()
    p = plan_with(mix={"kind": "rule", "rule": "standard", "scale": 3.0, "knobs": {}, "space": "auto"})
    assert close(run(p, x, c, u, sig, w=9.0), cfg(c, u, 3.0))


def test_zero_star_zero_init_leaves_latent():
    x, c, u, sig = batch(14.6)
    p = plan_with(mix={"kind": "rule", "rule": "cfg_zero_star", "scale": -1.0, "knobs": {"zero_init_steps": 1},
                       "space": "auto"})
    assert close(run(p, x, c, u, sig), x, 1e-4)


def test_apg_momentum_resets_on_new_run():
    x, c, u, _ = batch()
    p = plan_with(mix={"kind": "rule", "rule": "apg", "scale": -1.0,
                       "knobs": {"eta": 0.0, "norm_threshold": 0.0, "momentum": -0.5}, "space": "x0"})
    rt = engine.GuidanceRuntime(p)
    mo = {"transformer_options": {"sample_sigmas": torch.tensor([14.6, 6.0, 2.0, 0.5, 0.0])}}
    first = rt.guided_x0(x, c, u, torch.full((B,), 6.0), 6.0, model=FakeModel(), model_options=mo)
    rt.guided_x0(x, c, u, torch.full((B,), 2.0), 6.0, model=FakeModel(), model_options=mo)
    again = rt.guided_x0(x, c, u, torch.full((B,), 6.0), 6.0, model=FakeModel(), model_options=mo)
    assert close(first, again, 1e-5), "a rising sigma must reset the momentum"


def test_formula_equals_cfg_in_every_space():
    x, c, u, sig = batch()
    for space in engine.SPACES:
        p = plan_with(mix={"kind": "formula", "formula": "u + w * (c - u)", "space": space, "scale": -1.0})
        assert close(run(p, x, c, u, sig), cfg(c, u, 6.0), 5e-4), space


def test_space_round_trip_is_exact_enough():
    # float64 conversions: plain CFG computed in eps or v space lands on the x0-space result to float32 precision
    x, c, u, sig = batch(5.0)
    ref = run(plan_with(mix={"kind": "formula", "formula": "u + w * (c - u)", "space": "x0", "scale": -1.0}), x, c, u, sig)
    for space in ("eps", "v"):
        out = run(plan_with(mix={"kind": "formula", "formula": "u + w * (c - u)", "space": space, "scale": -1.0}),
                  x, c, u, sig)
        assert float((out - ref).abs().max()) <= 4 * torch.finfo(torch.float32).eps * float(ref.abs().max()), space


def test_formula_helpers_and_multiline():
    x, c, u, sig = batch()
    f = "d = c - u\nresult = u + w * orth(d, c) + proj(d, c)"
    p = plan_with(mix={"kind": "formula", "formula": f, "space": "x0", "scale": -1.0})
    out = run(p, x, c, u, sig)
    d = c - u
    par = (d * c).flatten(1).sum(1).view(B, 1, 1, 1) / c.flatten(1).pow(2).sum(1).view(B, 1, 1, 1) * c
    assert close(out, u + 6.0 * (d - par) + par, 5e-4)


def test_formula_errors_are_readable():
    try:
        engine.compile_formula("u + w * (c - u")
    except ValueError as e:
        assert "syntax" in str(e)
    else:
        raise AssertionError("expected a syntax error")
    x, c, u, sig = batch()
    p = plan_with(mix={"kind": "formula", "formula": "u + w * nope", "space": "x0", "scale": -1.0})
    try:
        run(p, x, c, u, sig)
    except ValueError as e:
        assert "nope" in str(e) and "formula" in str(e)
    else:
        raise AssertionError("expected a name error")


def test_schedule_constant_and_window():
    x, c, u, sig = batch(2.0)
    p = plan_with(when={"shape": "constant", "start": 0.0, "end": 1.0, "outside": "cond", "floor": 0.0})
    assert close(run(p, x, c, u, sig), cfg(c, u, 6.0))
    # window ends before sigma 2.0 is reached (end percent maps to a sigma above 2.0): outside
    p = plan_with(when={"shape": "constant", "start": 0.0, "end": 0.2, "outside": "cond", "floor": 0.0})
    assert close(run(p, x, c, u, sig), c)
    p["when"]["outside"] = "base"
    assert close(run(p, x, c, u, sig), cfg(c, u, 6.0))
    p["when"]["outside"] = "fixed"
    p["when"]["outside_scale"] = 2.0
    assert close(run(p, x, c, u, sig), cfg(c, u, 2.0))


def test_schedules_keep_the_average():
    for shape in ("linear_up", "linear_down", "cosine_up", "cosine_down", "v_shape", "lambda_shape"):
        vals = [engine.schedule_value(shape, 6.0, i / 400, 1 - i / 400) for i in range(401)]
        assert abs(sum(vals) / len(vals) - 6.0) < 0.05, (shape, sum(vals) / len(vals))
    assert engine.schedule_value("constant", 6.0, 0.3, 0.5) == 6.0


def test_bands_and_region():
    x, c, u, sig = batch()
    ref = cfg(c, u, 6.0)
    for method in ("gaussian", "fft"):
        p = plan_with(bands={"method": method, "low": 1.0, "high": 1.0, "blur_sigma": 2.0, "fft_cutoff": 0.3})
        assert close(run(p, x, c, u, sig), ref), method
        p["bands"].update(low=0.5, high=0.5)
        assert close(run(p, x, c, u, sig), cfg(c, u, 1 + 0.5 * 5.0)), method
        p["bands"].update(low=0.0, high=1.0)
        out = run(p, x, c, u, sig)
        low, high = engine.split_bands(ref - c, method, 2.0, 0.3)
        assert close(out, c + high), method
    mask = torch.ones(1, 64, 64)
    p = plan_with(region={"mask": mask, "inside": 1.0, "outside": 1.0, "feather": 0.0, "invert": False})
    assert close(run(p, x, c, u, sig), ref)
    p["region"]["inside"] = 2.0
    assert close(run(p, x, c, u, sig), cfg(c, u, 1 + 2 * 5.0))
    p["region"]["invert"] = True
    p["region"]["outside"] = 0.0
    assert close(run(p, x, c, u, sig), c)


def test_corrections():
    x, c, u, sig = batch()
    ref = cfg(c, u, 6.0)
    for method in engine.CORRECTIONS:
        p = plan_with(correct=[{"method": method, "strength": 0.0, "space": "auto"}])
        assert close(run(p, x, c, u, sig), ref), method
    p = plan_with(correct=[{"method": "rescale_std", "strength": 1.0, "space": "x0"}])
    out = run(p, x, c, u, sig)
    assert close(out.flatten(1).std(1), c.flatten(1).std(1), 1e-4)
    p = plan_with(correct=[{"method": "norm_cap", "strength": 1.0, "space": "x0", "cap_ratio": 1.05}])
    out = run(p, x, c, u, sig)
    assert bool((out.flatten(1).norm(dim=1) <= 1.05 * c.flatten(1).norm(dim=1) * (1 + 1e-4)).all())
    p = plan_with(correct=[{"method": "energy_preserve", "strength": 1.0, "space": "x0"}])
    out = run(p, x, c, u, sig)
    assert close(out.flatten(1).norm(dim=1), c.flatten(1).norm(dim=1), 1e-4)
    p = plan_with(correct=[{"method": "channel_norm_match", "strength": 1.0, "space": "x0"}])
    out = run(p, x, c, u, sig)
    assert close(torch.linalg.vector_norm(out, dim=1), torch.linalg.vector_norm(c, dim=1), 1e-4)
    # stacking applies in order
    p = plan_with(correct=[{"method": "rescale_std", "strength": 1.0, "space": "x0"},
                           {"method": "energy_preserve", "strength": 1.0, "space": "x0"}])
    out = run(p, x, c, u, sig)
    assert close(out.flatten(1).norm(dim=1), c.flatten(1).norm(dim=1), 1e-4)


def test_three_way_rules():
    x, c, u, sig = batch()
    n = u + 0.2 * torch.randn_like(u)
    p = engine.empty_plan()
    p["three_way_space"] = "x0"
    out = run(p, x, c, u, sig, x0_n=n, three_way_rule="negative_as_null")
    assert close(out, cfg(c, n, 6.0))
    out = run(p, x, c, u, sig, x0_n=n, three_way_rule="separate_negative", neg_scale=0.0)
    assert close(out, cfg(c, u, 6.0))
    out = run(p, x, c, u, sig, x0_n=n, three_way_rule="perp_neg", neg_scale=1.0)
    pos, neg = c - u, n - u
    coef = (pos * neg).flatten(1).sum(1).view(B, 1, 1, 1) / pos.flatten(1).pow(2).sum(1).view(B, 1, 1, 1)
    assert close(out, u + 6.0 * (pos - (neg - coef * pos)))


def test_weak_add_and_replace_without_comfy_pass():
    # the perturbed pass itself needs ComfyUI; check the combination through a stubbed perturbed_prediction
    x, c, u, sig = batch()
    pert = c - 0.1 * torch.randn_like(c)
    orig = engine.perturbed_prediction
    engine.perturbed_prediction = lambda *a, **k: pert
    try:
        p = plan_with(weak={"method": "pag", "scale": 3.0, "mode": "add", "blocks": [("middle", 0)]})
        out = run(p, x, c, u, sig, input_cond=[{}])
        assert close(out, cfg(c, u, 6.0) + 3.0 * (c - pert))
        p["weak"]["mode"] = "replace"
        out = run(p, x, c, u, sig, input_cond=[{}])
        assert close(out, cfg(c, pert, 6.0))
    finally:
        engine.perturbed_prediction = orig


def test_blur_tokens_and_patch_shapes():
    q = torch.randn(2, 64, 32)
    same = engine._blur_tokens(q, [2, 8, 8, 8], 0.1)
    assert same.shape == q.shape and close(same, q, 1e-3)
    inf = engine._blur_tokens(q, [2, 8, 8, 8], 100.0)
    assert close(inf, q.mean(1, keepdim=True).expand_as(q))
    pag = engine.make_attention_patch("pag")
    v = torch.randn(2, 64, 32)
    assert pag(q, q, v, {"n_heads": 4}) is v


def test_dtype_preserved():
    x, c, u, sig = batch()
    for dt in (torch.float16, torch.bfloat16):
        out = run(plan_with(mix={"kind": "rule", "rule": "apg", "scale": -1.0,
                                 "knobs": {"eta": 0.0, "norm_threshold": 15.0, "momentum": 0.0}, "space": "auto"}),
                  x.to(dt), c.to(dt), u.to(dt), sig)
        assert out.dtype == dt and torch.isfinite(out.float()).all()


def test_probe_writes_lines():
    x, c, u, sig = batch()
    with tempfile.TemporaryDirectory() as d:
        p = plan_with(measure={"prefix": "t", "print_every": 0, "folder": d})
        rt = engine.GuidanceRuntime(p)
        mo = {"transformer_options": {"sample_sigmas": torch.tensor([14.6, 6.0, 2.0, 0.5, 0.0])}}
        for s in (14.6, 6.0, 2.0, 0.5):
            rt.guided_x0(x, c, u, torch.full((B,), s), 6.0, model=FakeModel(), model_options=mo)
        files = os.listdir(d)
        assert len(files) == 1, files
        rows = [json.loads(line) for line in open(os.path.join(d, files[0]), encoding="utf-8")]
        assert "header" in rows[0] and len(rows) == 5
        for key in ("w", "push_ratio", "std_ratio", "delta_rms", "cos_c_u", "parallel_share", "lowfreq_share", "step", "sigma"):
            assert key in rows[1], key
        assert [r["step"] for r in rows[1:]] == [0, 1, 2, 3]


def test_uncond_skip_function():
    p = plan_with(when={"shape": "constant", "start": 0.0, "end": 0.2, "outside": "cond", "skip_uncond": True})
    rt = engine.GuidanceRuntime(p)
    assert rt.may_skip_uncond()
    seen = []

    def previous(args):
        seen.append(args["conds"][1])
        return ["c", "u"]
    fn = rt.make_calc_cond_batch_function(previous)
    fn({"conds": ["C", "U"], "input": None, "sigma": torch.tensor([14.0]), "model": FakeModel(), "model_options": {}})
    fn({"conds": ["C", "U"], "input": None, "sigma": torch.tensor([0.5]), "model": FakeModel(), "model_options": {}})
    assert seen == ["U", None], seen
    # another node's post-CFG function reads the unconditional: it is then always computed
    fn({"conds": ["C", "U"], "input": None, "sigma": torch.tensor([0.5]), "model": FakeModel(),
        "model_options": {"sampler_post_cfg_function": [lambda a: a["denoised"]]}})
    assert seen == ["U", None, "U"], seen


GRID = torch.tensor([14.6, 10.0, 6.0, 4.0, 2.0, 1.0, 0.5, 0.2, 0.0])     # 8 steps; end 0.3 maps to sigma 2.28


def run_at(plan, sigma, x, c, u, w=6.0):
    rt = engine.GuidanceRuntime(plan)
    mo = {"transformer_options": {"sample_sigmas": GRID}}
    out = rt.guided_x0(x, c, u, torch.full((B,), sigma), w, model=FakeModel(), model_options=mo)
    return out, rt.last["w"]


def test_schedule_spans_the_window():
    x, c, u, _ = batch()
    p = plan_with(when={"shape": "linear_down", "start": 0.0, "end": 0.3, "outside": "cond", "floor": 0.0})
    assert abs(run_at(p, 14.6, x, c, u)[1] - 11.0) < 1e-9       # the window's first step: the shape's start
    assert abs(run_at(p, 4.0, x, c, u)[1] - 1.0) < 1e-9         # the window's last step: the shape's end
    whole = plan_with(when={"shape": "linear_down", "start": 0.0, "end": 1.0, "outside": "cond", "floor": 0.0})
    assert abs(run_at(whole, 4.0, x, c, u)[1] - (1 + 10 * (1 - 3 / 7))) < 1e-9   # whole-run window: unchanged


def test_outside_the_window_the_other_nodes_still_apply():
    x, c, u, _ = batch()
    chain = {"mix": {"kind": "rule", "rule": "angle_limit", "scale": -1.0, "knobs": {"max_angle": 0.3}, "space": "auto"},
             "bands": {"method": "gaussian", "low": 1.0, "high": 3.0, "blur_sigma": 2.0, "fft_cutoff": 0.25}}
    when = {"shape": "cosine_down", "start": 0.0, "end": 0.3, "floor": 0.0, "outside_scale": 2.0}
    at = 2.0                                                    # outside the window (sigma below 2.28)
    ref6, _ = run_at(plan_with(**chain), at, x, c, u)
    ref2, _ = run_at(plan_with(**chain), at, x, c, u, w=2.0)
    out, _ = run_at(plan_with(when=dict(when, outside="base"), **chain), at, x, c, u)
    assert torch.equal(out, ref6)
    out, _ = run_at(plan_with(when=dict(when, outside="fixed"), **chain), at, x, c, u)
    assert torch.equal(out, ref2)
    out, _ = run_at(plan_with(when=dict(when, outside="plain"), **chain), at, x, c, u)
    assert close(out, cfg(c, u, 6.0))
    out, _ = run_at(plan_with(when=dict(when, outside="cond"), **chain), at, x, c, u)
    assert torch.equal(out, c)
    # the weak branch replacing the unconditional (the unconditional pass skipped): the chain still runs outside
    weak_replace = {"method": "pag", "scale": 2.0, "mode": "replace", "blocks": [("middle", 0)]}
    rt = engine.GuidanceRuntime(plan_with(when=dict(when, outside="base"), weak=weak_replace, **chain))
    assert rt._uncond_unused(torch.tensor([at]), FakeModel())
    captured = {}
    original = engine.perturbed_prediction
    engine.perturbed_prediction = lambda weak, model, ic, x_, s_, mo: captured.setdefault("x0_w", u)
    try:
        out = rt.guided_x0(x, c, None, torch.full((B,), at), 6.0, model=FakeModel(), input_cond=["cond"],
                           model_options={"transformer_options": {"sample_sigmas": GRID}}, uncond_valid=False)
    finally:
        engine.perturbed_prediction = original
    assert "x0_w" in captured and not torch.equal(out, c)


class StubPatcher:
    """The parts of ComfyUI's ModelPatcher that install() and clear() touch."""

    def __init__(self):
        self.model_options = {"transformer_options": {}}

    def set_model_sampler_cfg_function(self, fn, disable_cfg1_optimization=False):
        self.model_options["sampler_cfg_function"] = fn
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True

    def set_model_sampler_calc_cond_batch_function(self, fn):
        self.model_options["sampler_calc_cond_batch_function"] = fn


def test_install_compose_and_clear():
    m = StubPatcher()
    p = engine.read_plan(m)
    p["mix"] = {"kind": "rule", "rule": "apg", "scale": -1.0, "knobs": {}, "space": "auto"}
    engine.install(m, p)
    p2 = engine.read_plan(m)
    p2["when"] = {"shape": "constant", "start": 0.0, "end": 0.5, "outside": "cond", "skip_uncond": True}
    engine.install(m, p2)
    assert m.model_options[engine.PLAN_KEY]["mix"]["rule"] == "apg"
    assert m.model_options["disable_cfg1_optimization"] is True
    assert getattr(m.model_options["sampler_calc_cond_batch_function"], "_cfg_prototypes", False)
    text = engine.describe_plan(m.model_options[engine.PLAN_KEY])
    assert "apg" in text and "1 when" in text and "7 measure" in text
    assert p["when"] is None, "reading a plan must copy it"
    post = m.model_options["sampler_post_cfg_function"]
    assert len(post) == 1 and engine._ours(post[0]) and not engine.foreign_cfg_hooks(m.model_options)
    engine.clear(m)
    assert engine.PLAN_KEY not in m.model_options and "sampler_cfg_function" not in m.model_options
    assert "sampler_calc_cond_batch_function" not in m.model_options
    assert "sampler_post_cfg_function" not in m.model_options


def test_post_cfg_hands_on_the_exact_result():
    # ComfyUI's cfg_function computes x - fn(args), then runs the post-CFG functions in list order
    seen = []

    def theirs(args):                     # another node's post-CFG function, chained before this pack's nodes
        seen.append(args["denoised"])
        return args["denoised"]
    m = StubPatcher()
    m.model_options["sampler_post_cfg_function"] = [theirs]
    p = engine.read_plan(m)
    p["mix"] = {"kind": "rule", "rule": "standard", "scale": -1.0, "knobs": {}, "space": "auto"}
    engine.install(m, p)
    engine.install(m, engine.read_plan(m))                  # a second pack node: still one of ours, still first
    post = m.model_options["sampler_post_cfg_function"]
    assert len(post) == 2 and engine._ours(post[0]) and post[1] is theirs
    assert engine.foreign_cfg_hooks(m.model_options) and not engine.foreign_cfg_hooks({"sampler_post_cfg_function": post[:1]})
    torch.manual_seed(4)
    x = 1000.0 + torch.randn(1, 4, 8, 8)                    # a large latent: x - (x - x0) rounds at this scale
    c, u = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
    args = {"input": x, "cond_denoised": c, "uncond_denoised": u, "sigma": torch.tensor([5.0]), "cond_scale": 6.0,
            "model": FakeModel(), "model_options": m.model_options, "input_cond": None, "input_uncond": [{}]}
    fn = m.model_options["sampler_cfg_function"]
    denoised = x - fn(args)
    exact = fn.__self__._exact[1]
    assert close(exact, cfg(c, u, 6.0)) and not torch.equal(denoised, exact), "the control: the round trip must round here"
    for f in post:
        denoised = f({"denoised": denoised, "input": x})
    assert torch.equal(denoised, exact) and torch.equal(seen[-1], exact)
    other = post[0]({"denoised": denoised * 2, "input": x.clone()})     # nothing stored for this input: unchanged
    assert torch.equal(other, denoised * 2)
    engine.clear(m)
    assert m.model_options["sampler_post_cfg_function"] == [theirs]


def test_flow_noise_space_holds_at_sigma_one():
    # a flow schedule starts at sigma 1 (er_sde offsets it to 0.99997): the noise space stays an exact inverse there
    torch.manual_seed(3)
    x0 = torch.randn(B, 16, H, W, dtype=torch.float64)
    for sigma in (1.0, 1 - 3.3e-5, 0.9999, 0.5):
        x = (1 - sigma) * x0 + sigma * torch.randn_like(x0)
        sv = torch.full((B, 1, 1, 1), sigma, dtype=torch.float64)
        back = engine.from_space(engine.to_space(x0, x, sv, "eps", True), x, sv, "eps", True)
        assert close(back, x0, 1e-9), sigma
    x, c, u, _ = flow_batch()
    for sigma in (1.0, 1 - 3.3e-5):
        out = run_flow(formula_plan("u + w * (c - u)"), x, c, u, torch.full((B,), sigma))
        assert close(out, cfg(c, u, 4.5), 5e-4), sigma


def test_partial_window_mixes_per_sample():
    x, c, u, _ = batch()
    sig = torch.tensor([10.0, 0.5])
    p = plan_with(when={"shape": "constant", "start": 0.0, "end": 0.5, "outside": "cond", "skip_uncond": True})
    out = run(p, x, c, u, sig)
    assert close(out[0], cfg(c, u, 6.0)[0]) and close(out[1], c[1])


# ---------------------------------------------------------------------------- the governor (angle band)

UNITS = ("image", "pixel", "channel")


def gov(min_deg=0.0, max_deg=180.0, unit="image", home="cond", space="auto", start=0.0, end=1.0):
    return {"min_deg": min_deg, "max_deg": max_deg, "unit": unit, "home": home, "space": space, "start": start, "end": end}


def test_govern_neutral_band_is_bit_identical():
    x, c, u, sig = batch()
    plain = run(engine.empty_plan(), x, c, u, sig)
    for unit in UNITS:
        for space in ("auto", "eps", "v"):
            out = run(plan_with(govern=gov(0.0, 180.0, unit, space=space)), x, c, u, sig)
            assert torch.equal(out, plain), (unit, space)


def test_govern_ceiling_lands_on_the_bound_and_keeps_length():
    x, c, u, sig = batch()
    plain = run(engine.empty_plan(), x, c, u, sig)
    for unit in UNITS:
        before = engine._angle_deg(plain, c, unit)
        cap = float(before.median())                      # about half the units violate
        out = run(plan_with(govern=gov(0.0, cap, unit)), x, c, u, sig)
        after = engine._angle_deg(out, c, unit)
        fired = before > cap
        assert bool(fired.any()) and bool((~fired).any()), unit
        assert float((after[fired] - cap).abs().max()) < 1e-3, (unit, float((after[fired] - cap).abs().max()))
        assert float(after.max()) <= cap + 1e-3, unit
        n0 = engine._to_units(plain, unit).double().norm(dim=1)
        n1 = engine._to_units(out, unit).double().norm(dim=1)
        assert float(((n1 - n0).abs() / n0).max()) < 1e-5, unit
        same = engine._to_units(out, unit)[~fired] == engine._to_units(plain, unit)[~fired]
        assert bool(same.all()), unit                      # units inside the band come back bit for bit


def test_govern_turn_stays_in_the_plane_of_d_and_home():
    x, c, u, sig = batch()
    plain = run(engine.empty_plan(), x, c, u, sig)
    out = run(plan_with(govern=gov(0.0, 3.0, "image")), x, c, u, sig)
    for b in range(B):
        basis, _ = torch.linalg.qr(torch.stack([plain[b].flatten(), c[b].flatten()], 1).double())
        o = out[b].flatten().double()
        resid = o - basis @ (basis.T @ o)
        assert float(resid.norm() / o.norm()) < 1e-5


def test_govern_floor_turns_weak_guidance_and_skips_none():
    x, c, u, sig = batch()
    weak_u = c + 0.01 * (u - c)                            # guidance with a tiny angle
    plain = run(engine.empty_plan(), x, c, weak_u, sig, w=1.5)
    before = engine._angle_deg(plain, c, "image")
    floor = float(before.max()) + 5.0
    out = run(plan_with(govern=gov(floor, 180.0, "image")), x, c, weak_u, sig, w=1.5)
    assert float((engine._angle_deg(out, c, "image") - floor).abs().max()) < 1e-3
    same = run(plan_with(govern=gov(floor, 180.0, "image")), x, c, c.clone(), sig, w=1.5)   # no guidance at all
    assert torch.equal(same, run(engine.empty_plan(), x, c, c.clone(), sig, w=1.5))


def test_govern_only_offenders_move():
    x, c, u, sig = batch()
    u2 = u.clone()
    u2[1] = c[1] + 0.02 * (u[1] - c[1])                     # sample 1: small angle, inside the band
    plain = run(engine.empty_plan(), x, c, u2, sig)
    ang = engine._angle_deg(plain, c, "image")
    cap = float(ang.min()) + 0.5 * float(ang.max() - ang.min())
    out = run(plan_with(govern=gov(0.0, cap, "image")), x, c, u2, sig)
    assert torch.equal(out[1], plain[1]) and not torch.equal(out[0], plain[0])


def test_govern_unconditional_home_and_window():
    x, c, u, sig = batch()
    plain = run(engine.empty_plan(), x, c, u, sig)
    to_u = engine._angle_deg(plain, u, "image")
    cap = float(to_u.min()) * 0.5
    out = run(plan_with(govern=gov(0.0, cap, "image", home="uncond")), x, c, u, sig)
    assert float(engine._angle_deg(out, u, "image").max()) <= cap + 1e-3
    late_only = plan_with(govern=gov(0.0, cap, "image", home="uncond", start=0.9, end=1.0))
    assert torch.equal(run(late_only, x, c, u, torch.full((B,), 14.6)), run(engine.empty_plan(), x, c, u, torch.full((B,), 14.6)))
    assert not torch.equal(run(late_only, x, c, u, torch.full((B,), 0.05)), run(engine.empty_plan(), x, c, u, torch.full((B,), 0.05)))


def test_govern_probe_columns_and_readout():
    x, c, u, sig = batch()
    with tempfile.TemporaryDirectory() as d:
        p = plan_with(govern=gov(0.0, 5.0, "pixel"), measure={"prefix": "g", "print_every": 0, "folder": d})
        run(p, x, c, u, sig)
        rows = [json.loads(line) for line in open(os.path.join(d, os.listdir(d)[0]), encoding="utf-8")]
        r = rows[1]
        for key in ("gov_ceiling_share", "gov_floor_share", "gov_angle_mean_deg", "gov_angle_max_deg",
                    "angle_hat_c_deg", "angle_hat_c_pixel_p50_deg", "angle_hat_c_pixel_p95_deg", "angle_c_u_deg"):
            assert key in r, key
        assert 0.0 < r["gov_ceiling_share"] <= 1.0 and r["gov_floor_share"] == 0.0
        assert r["angle_hat_c_pixel_p95_deg"] <= 5.0 + 1e-3     # the probe reads the governed prediction
    text = engine.describe_plan(p)
    assert "6 govern: angle to the conditional held in [0, 5] degrees" in text and "7 measure" in text
    assert "6 govern: off" in engine.describe_plan(engine.empty_plan())


# -- flow matching with a shift, Anima's single-frame latents, the DiT weak branch, the formula box -------------------

class CONST:
    """Named like ComfyUI's CONST so is_flow_sampling reads the fakes below as flow sampling."""


class FakeFlowSampling(CONST):
    """ComfyUI's ModelSamplingDiscreteFlow (SD3, AuraFlow, Anima): sigma = s t / (1 + (s - 1) t), percent in raw t."""
    shift = 3.0

    def percent_to_sigma(self, p):
        if p <= 0:
            return 1.0
        if p >= 1:
            return 0.0
        t = 1.0 - p
        return self.shift * t / (1 + (self.shift - 1) * t)


class ModelSamplingFlux(CONST):
    """ComfyUI's Flux-type sampling: sigma = exp(mu) / (exp(mu) + 1/t - 1), mu stored as its shift."""
    shift = 1.15

    def percent_to_sigma(self, p):
        if p <= 0:
            return 1.0
        if p >= 1:
            return 0.0
        return math.exp(self.shift) / (math.exp(self.shift) + 1 / (1.0 - p) - 1)


class FakeFlowModel:
    model_sampling = FakeFlowSampling()


def flow_batch(t=0.4, shift=3.0, channels=16, frame=False):
    sigma = shift * t / (1 + (shift - 1) * t)
    x0_c = torch.randn(B, channels, H, W)
    x0_u = x0_c * 0.7 + 0.3 * torch.randn(B, channels, H, W)
    x = (1 - sigma) * x0_c + sigma * torch.randn(B, channels, H, W)
    if frame:
        x, x0_c, x0_u = x.unsqueeze(2), x0_c.unsqueeze(2), x0_u.unsqueeze(2)
    return x, x0_c, x0_u, torch.full((B,), sigma)


def run_flow(plan, x, c, u, sig, w=4.5, model=None, **kw):
    rt = engine.GuidanceRuntime(plan)
    mo = {"transformer_options": {"sample_sigmas": torch.tensor([1.0, 0.9, 0.75, 0.5, 0.25, 0.0])}}
    return rt.guided_x0(x, c, u, sig, w, model=model or FakeFlowModel(), model_options=mo, **kw)


def formula_plan(text, space="eps"):
    return plan_with(mix={"kind": "formula", "formula": text, "space": space, "scale": -1.0})


def test_flow_formula_variables_undo_the_shift():
    x, c, u, sig = flow_batch(t=0.4, shift=3.0)
    s = float(sig[0])
    check = (f"assert flow and shift == 3.0 and space == 'eps'\n"
             f"assert abs(t_raw - 0.4) < 1e-6, t_raw\n"
             f"assert abs(s_t - {s!r}) < 1e-6 and abs(a_t - (1 - {s!r})) < 1e-6\n"
             f"result = u + w * (c - u)")
    out = run_flow(formula_plan(check), x, c, u, sig)
    assert close(out, cfg(c, u, 4.5), 5e-4)
    # Flux-type sampling: the shift is exp(mu) and t_raw still undoes it
    assert abs(engine.flow_shift(ModelSamplingFlux()) - math.exp(1.15)) < 1e-12
    t = 0.3
    s_flux = math.exp(1.15) / (math.exp(1.15) + 1 / t - 1)
    flux = type("FluxModel", (), {"model_sampling": ModelSamplingFlux()})()
    out = run_flow(formula_plan("assert abs(t_raw - 0.3) < 1e-6, t_raw\nresult = c"), x, c, u,
                   torch.full((B,), s_flux), model=flux)
    assert close(out, c, 1e-4)
    # eps models: shift 1, t_raw = t, a_t = 1
    xe, ce, ue, se = batch(2.0)
    out = run(formula_plan("assert not flow and shift == 1.0 and a_t == 1.0 and t_raw == t\nresult = c"), xe, ce, ue, se)
    assert close(out, ce, 1e-4)


def test_flow_converters_read_noise_and_velocity():
    torch.manual_seed(1)
    sigma = 0.6
    x0, n = torch.randn(B, 16, H, W, dtype=torch.float64), torch.randn(B, 16, H, W, dtype=torch.float64)
    x = (1 - sigma) * x0 + sigma * n
    sv = torch.full((B, 1, 1, 1), sigma, dtype=torch.float64)
    assert close(engine.to_space(x0, x, sv, "eps", True), n, 1e-10)             # eps is the noise itself
    assert close(engine.to_space(x0, x, sv, "v", True), n - x0, 1e-10)          # v is the velocity
    env = engine.formula_flow_env("eps", x, sv, True, sigma, 3.0, 1 / 3)
    assert close(env["to_x0"](n), x0, 1e-10) and close(env["to_v"](n), n - x0, 1e-10) and close(env["to_eps"](n), n)
    assert close(env["from_x0"](x0), n, 1e-10) and close(env["from_v"](n - x0), n, 1e-10)       # and back
    assert env["a_t"] == 1 - sigma and env["s_t"] == sigma and env["flow"] is True
    # the README's example: plain CFG on the denoised image, written from the noise space, equals plain CFG
    x, c, u, sig = flow_batch()
    f = "from_x0(to_x0(u) + w * (to_x0(c) - to_x0(u)))"
    assert close(run_flow(formula_plan(f), x, c, u, sig), cfg(c, u, 4.5), 5e-4)


def test_single_frame_latents_run_as_images_and_come_back_5d():
    x5, c5, u5, sig = flow_batch(frame=True)
    plan = formula_plan("d = c - u\nresult = u + w * orth(d, c) + proj(d, c)")
    plan["govern"] = gov(0.0, 25.0, "pixel")
    out5 = run_flow(plan, x5, c5, u5, sig)
    out4 = run_flow(plan, x5.squeeze(2), c5.squeeze(2), u5.squeeze(2), sig)
    assert out5.shape == x5.shape and torch.equal(out5.squeeze(2), out4)


def test_dit_weak_branch_pieces():
    assert engine.dit_block_indices(28, "middle") == {13, 14}
    assert engine.dit_block_indices(28, "first third") == set(range(9))
    assert engine.dit_block_indices(28, "last third") == set(range(19, 28))
    assert engine.dit_block_indices(28, "all") == set(range(28))
    dit = types.SimpleNamespace(patch_spatial=2, patch_temporal=1, blocks=[None] * 28)
    assert engine.dit_token_grid(dit, torch.zeros(1, 16, 1, 33, 32)) == (1, 17, 16)     # padded up, then halved
    assert engine.dit_token_grid(dit, torch.zeros(1, 4, 32, 32)) is None
    q = torch.randn(2, 8 * 6, 32)
    inf = engine._blur_grid(q, 100.0, (1, 8, 6))
    assert close(inf, q.mean(1, keepdim=True).expand_as(q))                 # one frame: the frame mean
    two = engine._blur_grid(q, 100.0, (2, 4, 6)).reshape(2, 2, 24, 32)
    assert close(two, q.reshape(2, 2, 24, 32).mean(2, keepdim=True).expand_as(two))    # each frame its own mean
    ref = cv.gaussian_blur2d(q.reshape(2, 8, 6, 32).permute(0, 3, 1, 2), sigma=1.5).permute(0, 2, 3, 1).reshape(2, 48, 32)
    assert close(engine._blur_grid(q, 1.5, (1, 8, 6)), ref, 1e-5)
    assert engine._blur_grid(q, 1.5, (1, 7, 7)) is q                         # a grid that does not match: untouched
    patch = engine.make_dit_patch({3}, 100.0, (1, 8, 6))
    assert patch(q, q, q, pe=None, attn_mask=None, extra_options={"block_index": 2}) == {}
    assert close(patch(q, q, q, pe=None, attn_mask=None, extra_options={"block_index": 3})["q"], inf)
    # which methods run where
    unet = types.SimpleNamespace(diffusion_model=types.SimpleNamespace(input_blocks=[]))
    other = types.SimpleNamespace(diffusion_model=types.SimpleNamespace())
    assert engine.weak_support_error(unet, "pag") is None
    assert "neither" in engine.weak_support_error(other, "pag")
    orig = engine.dit_model
    engine.dit_model = lambda m: dit
    try:
        assert engine.weak_support_error(other, "seg") is None
        assert "Use SEG" in engine.weak_support_error(other, "pag")
        assert "Use SEG" in engine.weak_support_error(other, "temperature")
    finally:
        engine.dit_model = orig


def test_dit_weak_pass_adds_one_patch_and_leaves_the_options_alone():
    # the perturbed pass through stand-ins for ComfyUI's modules: the DiT gets one attn1_patch on the chosen blocks,
    # the caller's model options (and other nodes' patches in them) are not modified
    seen = {}

    def calc_cond_batch(model, conds, x, sigma, mo):
        seen["mo"] = mo
        return [x * 0.5]
    fake = {"comfy": types.ModuleType("comfy"), "comfy.model_patcher": types.ModuleType("comfy.model_patcher"),
            "comfy.samplers": types.ModuleType("comfy.samplers")}
    fake["comfy.samplers"].calc_cond_batch = calc_cond_batch
    fake["comfy"].samplers, fake["comfy"].model_patcher = fake["comfy.samplers"], fake["comfy.model_patcher"]
    saved = {k: sys.modules.get(k) for k in fake}
    orig = engine.dit_model
    dit = types.SimpleNamespace(patch_spatial=2, patch_temporal=1, blocks=[None] * 28)
    sys.modules.update(fake)
    engine.dit_model = lambda m: dit
    try:
        theirs = lambda *a, **k: {}                                          # noqa: E731 (another node's patch)
        mo = {"transformer_options": {"patches": {"attn1_patch": [theirs]}}}
        weak = {"method": "seg", "blur_sigma": 100.0, "blocks_label": "middle (PAG / SEG default)"}
        x = torch.randn(1, 16, 1, 16, 16)
        out = engine.perturbed_prediction(weak, object(), [{}], x, torch.tensor([0.5]), mo)
        assert torch.equal(out, x * 0.5)
        mine = seen["mo"]["transformer_options"]["patches"]["attn1_patch"]
        assert len(mine) == 2 and mine[0] is theirs
        assert mo["transformer_options"]["patches"]["attn1_patch"] == [theirs]     # the caller's options untouched
        q = torch.randn(1, 64, 8)                                                    # the (1, 8, 8) token grid
        assert mine[1](q, q, q, pe=None, attn_mask=None, extra_options={"block_index": 5}) == {}
        assert close(mine[1](q, q, q, pe=None, attn_mask=None, extra_options={"block_index": 13})["q"],
                     q.mean(1, keepdim=True).expand_as(q))
        try:
            engine.perturbed_prediction({"method": "pag"}, object(), [{}], x, torch.tensor([0.5]), mo)
        except ValueError as e:
            assert "Use SEG" in str(e)
        else:
            raise AssertionError("PAG on a DiT must raise")
    finally:
        engine.dit_model = orig
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


PENTA_V1 = """k = 1.0
r = 1 / math.sqrt(5)
V = torch.tensor([[1., 1., 1., -r], [1., -1., -1., -r], [-1., 1., -1., -r], [-1., -1., 1., -r],
                  [0., 0., 0., 4 * r]], dtype=c.dtype, device=c.device) * (math.sqrt(5) / 4)
g = (w - 1) * (c - u)
a = torch.einsum('kc,bchw->bkhw', V, g)
tau = k * a.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12
z = a / tau
m = z.abs().amax(dim=1, keepdim=True)
ep, en = torch.exp(z - m), torch.exp(-z - m)
result = c + 4 * tau * torch.einsum('kc,bkhw->bchw', V, ep - en) / (ep + en).sum(dim=1, keepdim=True)"""


def penta(c, u, w=7.0, k=None, text=None):
    text = text or open(os.path.join(ROOT, "formulas", "pentachoron.txt"), encoding="utf-8").read()
    if k is not None:
        assert text.count("k = 1.0 ") + text.count("k = 1.0\n") == 1
        text = text.replace("k = 1.0", f"k = {k}", 1)
    return engine.eval_formula(engine.compile_formula(text), {"c": c, "u": u, "w": w}, text)


def test_pentachoron_formula_on_4_and_16_channels():
    torch.manual_seed(2)
    c4 = torch.randn(2, 4, 16, 16, dtype=torch.float64)
    u4 = c4 + 0.15 * torch.randn_like(c4)
    assert close(penta(c4, u4), penta(c4, u4, text=PENTA_V1), 1e-12)        # the SDXL form is unchanged
    c16 = torch.randn(2, 16, 16, 16, dtype=torch.float64)
    u16 = c16 + 0.15 * torch.randn_like(c16)
    out = penta(c16, u16)
    assert out.shape == c16.shape and bool(torch.isfinite(out).all())
    assert close(penta(c16, u16, k=1e6), cfg(c16, u16, 7.0), 1e-6)        # a large k gives plain CFG back
    # each pixel's push in each group of 4 channels stays within 4 tau (tau = the image's RMS vertex coordinate)
    r = 1 / math.sqrt(5)
    V = torch.tensor([[1., 1., 1., -r], [1., -1., -1., -r], [-1., 1., -1., -r], [-1., -1., 1., -r], [0., 0., 0., 4 * r]],
                     dtype=torch.float64) * (math.sqrt(5) / 4)
    a = torch.einsum('kc,bgchw->bgkhw', V, (6.0 * (c16 - u16)).reshape(2, 4, 4, 16, 16))
    tau = a.pow(2).mean(dim=(1, 2, 3, 4)).sqrt().view(2, 1, 1, 1)
    push = (out - c16).reshape(2, 4, 4, 16, 16).norm(dim=2)
    assert bool((push <= 4 * tau * (1 + 1e-9)).all()) and float(push.max()) > 0
    try:
        penta(torch.randn(1, 6, 8, 8), torch.randn(1, 6, 8, 8))
    except ValueError as e:
        assert "divisible by 4" in str(e)
    else:
        raise AssertionError("6 channels must raise")
    x, c, u, sig = flow_batch(channels=16, frame=True)                     # through the engine on an Anima latent
    text = open(os.path.join(ROOT, "formulas", "pentachoron.txt"), encoding="utf-8").read()
    out = run_flow(formula_plan(text), x, c, u, sig)
    assert out.shape == x.shape and bool(torch.isfinite(out).all())


def test_formula_imports():
    x, c, u, sig = batch()
    f = "import torch.nn.functional as F\nresult = u + w * (F.relu(c - u) - F.relu(u - c))"
    assert close(run(formula_plan(f, "x0"), x, c, u, sig), cfg(c, u, 6.0), 5e-4)
    f = "t = tuple(range(1, c.ndim))\nresult = c + 0 * c.mean(dim=t, keepdim=True)"
    assert close(run(formula_plan(f, "x0"), x, c, u, sig), c, 1e-6)
    try:
        run(formula_plan("import os\nresult = c", "x0"), x, c, u, sig)
    except ValueError as e:
        assert "torch and numpy only" in str(e)
    else:
        raise AssertionError("import os must be refused")


def test_formula_builds_tensors_in_a_fresh_process():
    # torch's C code imports torch.storage the first time a tensor is built from a list, through the calling frame's
    # builtins; in a fresh process the formula's frame is that frame. The control (the import hook removed) must fail.
    code = ("import os, sys; os.environ['CUDA_VISIBLE_DEVICES'] = '-1'; sys.path.insert(0, {root!r}); import torch; "
            "from cfg_megapack import engine; {extra}c = torch.randn(1, 4, 8, 8); "
            "f = 'result = c + 0 * torch.tensor([[1., 2.]], dtype=c.dtype)[0, 0]'; "
            "print(tuple(engine.eval_formula(engine.compile_formula(f), {{'c': c, 'u': c, 'w': 1.0}}, f).shape))")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", HF_TOKEN="", HF_HUB_OFFLINE="1")
    ok = subprocess.run([sys.executable, "-c", code.format(root=ROOT, extra="")], capture_output=True, text=True,
                        timeout=600, env=env)
    assert ok.returncode == 0 and "(1, 4, 8, 8)" in ok.stdout, ok.stderr[-400:]
    control = subprocess.run([sys.executable, "-c", code.format(root=ROOT, extra="engine._SAFE_BUILTINS.pop('__import__'); ")],
                             capture_output=True, text=True, timeout=600, env=env)
    assert control.returncode != 0, "the control passed: this torch no longer imports lazily, the test proves nothing"


# -- the paper nodes (papers.py): every specification runs through the engine -----------------------------------

from cfg_megapack import papers  # noqa: E402


def _paper_plan(p, values=None, scale=-1.0):
    key, entry = papers.stage_entry(p, dict(values or {}), scale, engine.SPACE_LABELS[p.space])
    return plan_with(**{key: entry})


def test_every_combine_paper_runs_on_eps_and_flow_models():
    runs = 0
    for p in papers.PAPERS:
        if p.stage != "combine":
            continue
        for flow in (False, True):
            rt = engine.GuidanceRuntime(_paper_plan(p))
            if flow:
                x, c, u, _ = flow_batch()
                grid, model = [1.0, 0.8, 0.55, 0.3, 0.1, 0.0], FakeFlowModel()
            else:
                x, c, u, _ = batch()
                grid, model = [14.6, 6.0, 2.0, 0.5, 0.1, 0.0], FakeModel()
            mo = {"transformer_options": {"sample_sigmas": torch.tensor(grid)}}
            for s in grid[:-1]:                                 # several steps: the stateful ones update
                out = rt.guided_x0(x, c, u, torch.full((B,), s), 6.0, model=model, model_options=mo)
                assert out.shape == c.shape and bool(torch.isfinite(out).all()), (p.key, flow, s)
                runs += 1
    assert runs >= 2 * 5 * 30


def test_model_aware_paper_defaults():
    # SMC-CFG's k = -1 reaches the library as None and picks the size for the model kind
    x, c, u, _ = batch()
    smc = papers.BY_KEY["SMCCFG"]
    assert papers.knob_values(smc, {})["k"] is None
    for flow, k in ((False, cv.SMC_AUTO_K["eps"]), (True, cv.SMC_AUTO_K["flow"])):
        auto = cv.smc_cfg(c, u, 6.0, k=None, flow=flow)
        assert close(auto, cv.smc_cfg(c, u, 6.0, k=k), 1e-6), flow
    assert cv.SMC_AUTO_K == {"flow": 0.2, "eps": 0.01}
    # the paper plan hands the library the model kind: auto k on eps and flow models runs and differs from CFG
    for flow in (False, True):
        xs, cs, us, _ = flow_batch() if flow else batch()
        model = FakeFlowModel() if flow else FakeModel()
        rt = engine.GuidanceRuntime(_paper_plan(smc))
        out = rt.guided_x0(xs, cs, us, torch.full((B,), 0.5 if flow else 2.0), 4.5, model=model)
        assert bool(torch.isfinite(out).all()) and not close(out, cfg(cs, us, 4.5), 1e-6), flow
    # Adaptive Guidance reads the denoised predictions by default; FBG starts at the paper's Stable Diffusion values
    ag = papers.BY_KEY["AdaptiveGuidance"]
    assert engine.SPACE_LABELS[ag.space] == "x0" and _paper_plan(ag)["mix"]["space"] == "x0"
    assert {k.name: k.default for k in papers.BY_KEY["FBG"].knobs if k.name in ("pi", "t0", "t1")} == \
        {"pi": 0.85, "t0": 0.75, "t1": 0.5}


def test_paper_knobs_reach_the_library():
    # every knob's library keyword is accepted (apply_variant rejects unknown ones) and neutral settings give CFG
    x, c, u, sig = batch()
    neutral = {"RescaleCFG": {"phi": 0.0}, "APG": {"eta": 1.0, "norm_threshold": 0.0, "momentum": 0.0},
               "PowerLawCFG": {"alpha": 0.0}, "CFGRenorm": {"rho": 0.0}, "MAMBOG": {"alpha": 0.0},
               "SMCCFG": {"k": 0.0}, "PMCCFG": {"gamma_cap": 1000.0}, "VAGS": {"kappa": 0.0},
               "CFGOEC": {"tau": -1.0}, "EpsilonScaling": {"factor": 1.0}, "TSR": {"k": 1.0},
               "FDG": {"w_low": 6.0}, "FreSca": {"scale_low": 1.0, "scale_high": 1.0}, "LFCFG": {"rho": 1.0},
               "CFGZeroStar": {"zero_init_steps": 0}, "CFG": {}}
    for key, values in neutral.items():
        p = papers.BY_KEY[key]
        vals = {k.name: k.default for k in p.knobs}
        vals.update(values)
        out = run(_paper_plan(p, vals), x, c, u, sig)
        if key == "CFGZeroStar":                    # s* rescales u: CFG only when u already fits c
            continue
        assert close(out, cfg(c, u, 6.0), 2e-3), key
    for p in papers.PAPERS:
        if p.stage == "combine":
            params = engine._combiner_params(p.registry) | set(cv.REGISTRY[p.registry].get("state_knobs", ()))
            for k in p.knobs:
                assert (k.lib or k.name) in params, (p.key, k.name)


def test_when_papers_window_and_schedules():
    x, c, u, _ = batch()
    gi = papers.BY_KEY["GuidanceInterval"]
    plan = _paper_plan(gi, {"sigma_low": 0.28, "sigma_high": 5.42})
    for s, inside in ((10.0, False), (5.0, True), (0.5, True), (0.2, False)):
        out = run_at(plan, s, x, c, u)[0]
        assert close(out, cfg(c, u, 6.0) if inside else c, 1e-4), s
    # flow models see the bounds as noise levels sigma / (1 + sigma): (0.219, 0.844]
    xf, cf, uf, _ = flow_batch()
    rt = engine.GuidanceRuntime(plan)
    mo = {"transformer_options": {"sample_sigmas": torch.tensor([1.0, 0.9, 0.5, 0.1, 0.0])}}
    for s, inside in ((0.9, False), (0.8, True), (0.25, True), (0.2, False)):
        out = rt.guided_x0(xf, cf, uf, torch.full((B,), s), 4.5, model=FakeFlowModel(), model_options=mo)
        assert close(out, cfg(cf, uf, 4.5) if inside else cf, 1e-4), s
    assert "EDM units" in engine.describe_plan(plan)
    for key in ("WangSchedules", "TVCFG", "C2FG", "EarlyHighLateUncond", "CFGTruncation"):
        p = papers.BY_KEY[key]
        plan = _paper_plan(p)
        for s in (14.6, 6.0, 2.0, 0.5):
            out = run_at(plan, s, x, c, u)[0]
            assert bool(torch.isfinite(out).all()), key
    late = run_at(_paper_plan(papers.BY_KEY["EarlyHighLateUncond"], {"switch": 0.3, "late_scale": 0.0}), 0.5, x, c, u)[0]
    assert close(late, u, 1e-4)                                  # after the switch: the unconditional alone


def test_weak_papers_build_their_stage():
    for key, method in (("PAG", "pag"), ("SEG", "seg"), ("STG", "skip")):
        p = papers.BY_KEY[key]
        k, entry = papers.stage_entry(p, {})
        assert k == "weak" and entry["method"] == method and entry["mode"] == "add"
        assert entry["blocks"] == list(engine.BLOCK_PRESETS["middle (PAG / SEG default)"])
    dit = types.SimpleNamespace(patch_spatial=2, patch_temporal=1, blocks=[None] * 28)
    orig = engine.dit_model
    engine.dit_model = lambda m: dit
    try:
        assert engine.weak_support_error(object(), "skip") is None and engine.weak_support_error(object(), "pag")
    finally:
        engine.dit_model = orig
    q = torch.randn(1, 64, 8)
    patch = engine.make_dit_patch({13}, 10.0, (1, 8, 8), "skip")
    assert torch.equal(patch(q, q, q, extra_options={"block_index": 13})["v"], torch.zeros_like(q))
    assert patch(q, q, q, extra_options={"block_index": 12}) == {}
    skip = engine.make_skip_patch([("middle", 0)])
    n = torch.randn(2, 16, 8)
    assert torch.equal(skip(n, {"block": ("middle", 0)}), torch.zeros_like(n)) and skip(n, {"block": ("output", 0)}) is n


def test_guider_papers_run_through_the_three_way_path():
    x, c, u, sig = batch()
    n = u + 0.2 * torch.randn_like(u)
    for p in papers.PAPERS:
        if p.stage != "guider":
            continue
        rt = engine.GuidanceRuntime(plan_with())
        for s in (14.6, 6.0, 2.0):
            out = rt.guided_x0(x, c, u, torch.full((B,), s), 6.0, model=FakeModel(),
                               model_options={"transformer_options": {"sample_sigmas": torch.tensor([14.6, 6.0, 2.0, 0.0])}},
                               x0_n=n, three_way_rule="registry:" + p.registry,
                               three_way_knobs=papers.guider_knobs(p, {k.name: k.default for k in p.knobs}))
            assert bool(torch.isfinite(out).all()), p.key
    rt = engine.GuidanceRuntime(plan_with(three_way_space="x0"))
    out = rt.guided_x0(x, c, u, sig, 6.0, model=FakeModel(), x0_n=n, three_way_rule="registry:composable_not",
                       three_way_knobs={})
    assert close(out, u + 6.0 * (c - n), 1e-4)


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # report every test
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - failed} passed, {failed} failed out of {len(TESTS)} | torch {torch.__version__} | "
          f"cuda available: {torch.cuda.is_available()}")
    sys.exit(1 if failed else 0)
