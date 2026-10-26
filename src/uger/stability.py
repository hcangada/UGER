import math
from collections.abc import Mapping

import torch
import torch.nn.functional as functional

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call


def global_norm(values):
    tensors = list(values.values()) if isinstance(values, Mapping) else list(values)
    if not tensors:
        return torch.tensor(0.0)
    dtype = torch.float64 if any(t.dtype == torch.float64 for t in tensors) else torch.float32
    return torch.stack([tensor.to(dtype).square().sum() for tensor in tensors]).sum().sqrt()


def project_global(perturbation, radius):
    if radius < 0:
        raise ValueError("The projection radius must be nonnegative.")
    if not perturbation:
        return {}
    if radius == 0:
        return {name: torch.zeros_like(value) for name, value in perturbation.items()}
    norm = global_norm(perturbation)
    scale = radius / norm.clamp_min(radius)
    return {name: value * scale.to(value.dtype) for name, value in perturbation.items()}


def _top_items(items, scores, quota):
    if quota == 0:
        return items[:0]
    id_order = torch.argsort(items)
    score_order = torch.argsort(scores[id_order], descending=True, stable=True)
    return items[id_order[score_order[:quota]]]


def build_ranked_candidates(proposal_ids, scores, anchor, size):
    if proposal_ids.ndim != 1 or scores.shape != proposal_ids.shape:
        raise ValueError("Proposal items and scores must be matching one-dimensional tensors.")
    if int(size) != size or size < 1:
        raise ValueError("The candidate budget must be a positive integer.")
    size = int(size)
    if proposal_ids.unique().numel() != proposal_ids.numel():
        raise ValueError("Proposal items must be distinct.")
    anchor = int(anchor)
    if not bool((proposal_ids == anchor).any()):
        raise ValueError("The proposal pool must contain its positive anchor.")
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("Candidate scores must be finite.")
    if proposal_ids.numel() <= size:
        return torch.sort(proposal_ids).values
    alternatives = proposal_ids != anchor
    selected = _top_items(proposal_ids[alternatives], scores[alternatives], size - 1)
    return torch.sort(torch.cat((selected, proposal_ids.new_tensor([anchor])))).values


def build_candidates(proposal_ids, clean_scores, probe_scores, anchor, clean_quota, probe_quota):
    if proposal_ids.ndim != 1 or clean_scores.shape != proposal_ids.shape or probe_scores.shape != proposal_ids.shape:
        raise ValueError("Proposal items and their scores must be matching one-dimensional tensors.")
    if clean_quota < 0 or probe_quota < 0:
        raise ValueError("Candidate quotas must be nonnegative.")
    if any(int(value) != value for value in (clean_quota, probe_quota)):
        raise ValueError("Candidate quotas must be integers.")
    clean_quota, probe_quota = int(clean_quota), int(probe_quota)
    if not bool(torch.isfinite(clean_scores).all() and torch.isfinite(probe_scores).all()):
        raise FloatingPointError("Candidate scores must be finite.")
    if proposal_ids.unique().numel() != proposal_ids.numel():
        raise ValueError("Proposal items must be distinct.")
    anchor = int(anchor)
    if not bool((proposal_ids == anchor).any()):
        raise ValueError("The proposal pool must contain its positive anchor.")
    target = clean_quota + probe_quota + 1
    if proposal_ids.numel() <= target:
        return torch.sort(proposal_ids).values
    clean = _top_items(proposal_ids, clean_scores, clean_quota)
    remaining = ~torch.isin(proposal_ids, clean)
    probe = _top_items(proposal_ids[remaining], probe_scores[remaining], probe_quota)
    positive = proposal_ids.new_tensor([anchor])
    candidates = torch.unique(torch.cat((clean, probe, positive)), sorted=True)
    deficit = target - candidates.numel()
    if deficit:
        remaining = ~torch.isin(proposal_ids, candidates)
        filler = _top_items(proposal_ids[remaining], probe_scores[remaining], deficit)
        candidates = torch.sort(torch.cat((candidates, filler))).values
    return candidates


class UGERStability:
    build_candidates = staticmethod(build_candidates)

    def __init__(self, epsilon=0.01, proposal_size=256, clean_quota=20, probe_quota=20, nu=1e-12, fallback_scale=0.01, refine=True, candidate_policy="enhanced", initialization="incoming", candidate_probe="incoming", random_scale=0.01):
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("The perturbation radius must be finite and positive.")
        if proposal_size < 1 or clean_quota < 0 or probe_quota < 0:
            raise ValueError("Proposal size must be positive and candidate quotas nonnegative.")
        if any(int(value) != value for value in (proposal_size, clean_quota, probe_quota)):
            raise ValueError("Proposal size and candidate quotas must be integers.")
        if clean_quota + probe_quota + 1 > proposal_size:
            raise ValueError("The candidate budget cannot exceed the proposal budget.")
        if not math.isfinite(nu) or nu <= 0:
            raise ValueError("The normalization constant must be finite and positive.")
        if not math.isfinite(fallback_scale) or not 0 < fallback_scale <= 1:
            raise ValueError("The fallback scale must lie in (0, 1].")
        if not math.isfinite(random_scale) or not 0 < random_scale <= 1:
            raise ValueError("The random initialization scale must lie in (0, 1].")
        if candidate_policy not in {"enhanced", "clean", "probe"}:
            raise ValueError("The candidate policy must be enhanced, clean, or probe.")
        if initialization not in {"incoming", "random", "kl"}:
            raise ValueError("The perturbation initialization must be incoming, random, or kl.")
        if candidate_probe not in {"incoming", "random", "initial"}:
            raise ValueError("The candidate probe must be incoming, random, or initial.")
        self.epsilon = float(epsilon)
        self.proposal_size = int(proposal_size)
        self.clean_quota = int(clean_quota)
        self.probe_quota = int(probe_quota)
        self.nu = float(nu)
        self.fallback_scale = float(fallback_scale)
        self.refine = bool(refine)
        self.candidate_policy = candidate_policy
        self.initialization = initialization
        self.candidate_probe = candidate_probe
        self.random_scale = float(random_scale)

    def _generators(self, generator, device):
        seeds = torch.randint(0, 2**31 - 1, (4,), generator=generator, device=generator.device).cpu().tolist()
        return [torch.Generator(device=device).manual_seed(seed) for seed in seeds]

    def _random_perturbation(self, trainable, radius, generator):
        noise = {name: torch.randn(parameter.shape, dtype=parameter.dtype, device=parameter.device, generator=generator) for name, parameter in trainable.items()}
        norm = global_norm(noise)
        if bool(norm <= self.nu):
            nonempty = next((value for value in noise.values() if value.numel()), None)
            if nonempty is None:
                raise ValueError("The recommendation model has no nonempty trainable parameters.")
            nonempty.reshape(-1)[0] = 1.0
            norm = global_norm(noise)
        return {name: (value * (radius / norm).to(value.dtype)).detach() for name, value in noise.items()}

    def _scores(self, model, parameters, perturbation, users, items, graph, masks):
        state = {name: value + perturbation[name] if name in perturbation else value for name, value in parameters.items()}
        state.update({name: value.detach().clone() for name, value in model.named_buffers()})
        return functional_call(model, state, (users, items, graph), {"masks": masks})

    def _probe(self, model, incoming_batch, trainable, graph, generator, masks):
        names = list(trainable)
        parameters = list(trainable.values())
        gradients = [None] * len(parameters)
        if incoming_batch["users"].numel():
            loss = model.recommendation_loss(incoming_batch, graph, masks=masks)
            if loss.requires_grad:
                gradients = torch.autograd.grad(loss, parameters, allow_unused=True, create_graph=False)
        gradient = {name: torch.zeros_like(parameter) if value is None else value.detach() for name, parameter, value in zip(names, parameters, gradients)}
        norm = global_norm(gradient)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("The incoming recommendation gradient is not finite.")
        fallback = bool(norm <= self.nu)
        if fallback:
            probe = self._random_perturbation(trainable, self.fallback_scale * self.epsilon, generator)
        else:
            probe = {name: (-self.epsilon * value / (norm + self.nu).to(value.dtype)).detach() for name, value in gradient.items()}
        return probe, float(norm.detach().cpu()), fallback

    def _proposals(self, users, anchors, eligible_items, history, generator, device):
        eligible = torch.as_tensor(eligible_items, dtype=torch.long, device=device).reshape(-1).unique(sorted=True)
        eligible_list = eligible.detach().cpu().tolist()
        eligible_set = set(eligible_list)
        pools = []
        for user, anchor in zip(users.detach().cpu().tolist(), anchors.detach().cpu().tolist()):
            excluded = set(history.get(int(user), ()))
            excluded.add(int(anchor))
            remaining_count = len(eligible_set) - sum(item in eligible_set for item in excluded)
            quota = min(self.proposal_size - 1, remaining_count)
            selected = []
            selected_set = set()
            if quota and (remaining_count <= 4 * quota or remaining_count * 2 < len(eligible_set)):
                available = [item for item in eligible_list if item not in excluded]
                order = torch.randperm(len(available), generator=generator, device=device)[:quota].cpu().tolist()
                selected = [available[index] for index in order]
            elif quota:
                for _ in range(8):
                    needed = quota - len(selected)
                    if not needed:
                        break
                    indices = torch.randint(eligible.numel(), (max(32, 2 * needed),), generator=generator, device=device)
                    for item in eligible[indices].cpu().tolist():
                        if item not in excluded and item not in selected_set:
                            selected.append(item)
                            selected_set.add(item)
                            if len(selected) == quota:
                                break
                if len(selected) < quota:
                    available = [item for item in eligible_list if item not in excluded and item not in selected_set]
                    order = torch.randperm(len(available), generator=generator, device=device)[:quota - len(selected)].cpu().tolist()
                    selected.extend(available[index] for index in order)
            pools.append(torch.tensor([anchor, *selected], dtype=torch.long, device=device))
        return pools

    def _pack(self, rows, anchors):
        width = max(row.numel() for row in rows)
        items = anchors[:, None].expand(-1, width).clone()
        mask = torch.zeros_like(items, dtype=torch.bool)
        for index, row in enumerate(rows):
            items[index, :row.numel()] = row
            mask[index, :row.numel()] = True
        return items, mask

    def _discrepancy(self, model, parameters, perturbation, users, items, mask, reference, graph, masks):
        logits = self._scores(model, parameters, perturbation, users, items, graph, masks).to(torch.float64)
        log_probability = functional.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=-1).masked_fill(~mask, 0.0)
        return functional.kl_div(log_probability, reference, reduction="none").sum(dim=-1).mean().clamp_min(0.0)

    def _kl_gradient(self, model, parameters, perturbation, users, items, mask, reference, graph, masks):
        search = {name: value.detach().clone().requires_grad_(True) for name, value in perturbation.items()}
        discrepancy = self._discrepancy(model, parameters, search, users, items, mask, reference, graph, masks)
        if discrepancy.requires_grad:
            gradients = torch.autograd.grad(discrepancy, list(search.values()), allow_unused=True, create_graph=False)
        else:
            gradients = [None] * len(search)
        gradient = {name: torch.zeros_like(value) if grad is None else grad.detach() for (name, value), grad in zip(search.items(), gradients)}
        norm = global_norm(gradient)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("The consistency gradient is not finite.")
        return gradient, norm

    def compute(self, model, incoming_batch, replay_batch, graph, history, eligible_items, generator, masks=None, proposal_pools=None, candidate_sets=None):
        parameters = dict(model.named_parameters())
        trainable = {name: value for name, value in parameters.items() if value.requires_grad}
        if not trainable:
            raise ValueError("The recommendation model has no trainable parameters.")
        first = next(iter(trainable.values()))
        users = replay_batch["users"].to(device=first.device, dtype=torch.long)
        if not users.numel():
            zero = sum(parameter.sum() * 0.0 for parameter in trainable.values())
            return zero, {"probe_norm": 0.0, "perturbation_norm": 0.0, "incoming_gradient_norm": 0.0, "fallback_used": False, "selected_index": 0, "selected_kl": 0.0, "candidate_items": users.new_empty((0, 0)), "candidate_mask": torch.empty((0, 0), dtype=torch.bool, device=first.device), "clean_reference": first.new_empty((0, 0)), "trial_kl": first.new_zeros(1)}
        if masks is None and hasattr(model, "sample_masks"):
            masks = model.sample_masks(graph, generator)
        anchors = replay_batch["items"][:, 0].to(device=first.device, dtype=torch.long)
        base = {name: value.detach() for name, value in parameters.items()}
        probe_generator, proposal_generator, initial_generator, candidate_generator = self._generators(generator, first.device)
        zero = {name: torch.zeros_like(value) for name, value in trainable.items()}
        need_incoming = self.initialization == "incoming" or (self.candidate_policy != "clean" and self.candidate_probe in {"incoming", "random"})
        if need_incoming:
            probe, incoming_norm, fallback = self._probe(model, incoming_batch, trainable, graph, probe_generator, masks)
        else:
            probe, incoming_norm, fallback = zero, 0.0, False
        with torch.no_grad():
            if proposal_pools is None:
                proposals = self._proposals(users, anchors, eligible_items, history, proposal_generator, first.device)
            else:
                proposals = [torch.as_tensor(row, dtype=torch.long, device=first.device).reshape(-1).clone() for row in proposal_pools]
                if len(proposals) != users.numel():
                    raise ValueError("There must be one proposal pool per replayed interaction.")
                for row, user, anchor in zip(proposals, users.tolist(), anchors.tolist()):
                    if not row.numel() or row.numel() > self.proposal_size or row.unique().numel() != row.numel() or not bool((row == anchor).any()):
                        raise ValueError("Proposal pools must contain distinct items, their anchor, and respect the proposal budget.")
                    if any(int(item) in history.get(int(user), ()) and int(item) != anchor for item in row.tolist()):
                        raise ValueError("Proposal comparison items must be unobserved for the replay user.")
            proposal_items, proposal_mask = self._pack(proposals, anchors)
            clean_scores = self._scores(model, base, {}, users, proposal_items, graph, masks)
        if self.initialization == "incoming":
            initial = probe
        elif self.initialization == "random":
            initial = self._random_perturbation(trainable, self.random_scale * self.epsilon, initial_generator)
        else:
            seed = self._random_perturbation(trainable, self.random_scale * self.epsilon, initial_generator)
            with torch.no_grad():
                pool_reference = functional.softmax(clean_scores.to(torch.float64).masked_fill(~proposal_mask, -torch.inf), dim=-1).detach()
            gradient, norm = self._kl_gradient(model, base, seed, users, proposal_items, proposal_mask, pool_reference, graph, masks)
            initial = {name: (self.epsilon * value / (norm + self.nu).to(value.dtype)).detach() for name, value in gradient.items()}
        if self.candidate_probe == "initial":
            candidate_perturbation = initial
        elif self.candidate_probe == "random" and self.candidate_policy != "clean":
            candidate_perturbation = self._random_perturbation(trainable, global_norm(probe), candidate_generator)
        else:
            candidate_perturbation = probe
        with torch.no_grad():
            if self.candidate_policy == "clean":
                probe_scores = clean_scores
            else:
                probe_scores = self._scores(model, base, candidate_perturbation, users, proposal_items, graph, masks)
            target = self.clean_quota + self.probe_quota + 1
            candidates = []
            if candidate_sets is not None and len(candidate_sets) != len(proposals):
                raise ValueError("There must be one fixed candidate support per replay interaction.")
            for index, pool in enumerate(proposals):
                anchor = anchors[index].item()
                clean_row = clean_scores[index, :pool.numel()]
                probe_row = probe_scores[index, :pool.numel()]
                if candidate_sets is not None:
                    selected = torch.as_tensor(candidate_sets[index], dtype=torch.long, device=first.device).reshape(-1).clone()
                    if selected.unique().numel() != selected.numel() or selected.numel() != min(target, pool.numel()) or not bool((selected == anchor).any()) or not bool(torch.isin(selected, pool).all()):
                        raise ValueError("Fixed candidate supports must contain their anchor, lie within the pool, and match the candidate budget.")
                    selected = torch.sort(selected).values
                elif self.candidate_policy == "clean":
                    selected = build_ranked_candidates(pool, clean_row, anchor, target)
                elif self.candidate_policy == "probe":
                    selected = build_ranked_candidates(pool, probe_row, anchor, target)
                else:
                    selected = build_candidates(pool, clean_row, probe_row, anchor, self.clean_quota, self.probe_quota)
                candidates.append(selected)
            candidate_items, candidate_mask = self._pack(candidates, anchors)
            reference_logits = self._scores(model, base, {}, users, candidate_items, graph, masks).to(torch.float64)
            reference = functional.softmax(reference_logits.masked_fill(~candidate_mask, -torch.inf), dim=-1).detach()
        perturbations = [zero, initial]
        if self.refine:
            gradient, norm = self._kl_gradient(model, base, initial, users, candidate_items, candidate_mask, reference, graph, masks)
            refined = {name: initial[name] + self.epsilon * value / (norm + self.nu).to(value.dtype) for name, value in gradient.items()}
            perturbations.append({name: value.detach() for name, value in project_global(refined, self.epsilon).items()})
        with torch.no_grad():
            trials = torch.stack([self._discrepancy(model, base, perturbation, users, candidate_items, candidate_mask, reference, graph, masks) for perturbation in perturbations])
            selected = int(torch.argmax(trials).item()) if self.refine else 1
        chosen = {name: value.detach() for name, value in perturbations[selected].items()}
        loss = self._discrepancy(model, parameters, chosen, users, candidate_items, candidate_mask, reference, graph, masks)
        if not loss.requires_grad:
            loss = loss + sum(parameter.sum() * 0.0 for parameter in trainable.values())
        diagnostics = {"probe_norm": float(global_norm(probe).detach().cpu()), "perturbation_norm": float(global_norm(chosen).detach().cpu()), "incoming_gradient_norm": incoming_norm, "fallback_used": fallback, "selected_index": selected, "selected_kl": float(trials[selected].detach().cpu()), "candidate_items": candidate_items.detach(), "candidate_mask": candidate_mask.detach(), "clean_reference": reference.detach(), "trial_kl": trials.detach(), "proposal_items": proposal_items.detach(), "proposal_mask": proposal_mask.detach(), "clean_proposal_scores": clean_scores.detach(), "probe_proposal_scores": probe_scores.detach(), "probe_perturbation": {name: value.detach() for name, value in probe.items()}, "candidate_perturbation": {name: value.detach() for name, value in candidate_perturbation.items()}, "initial_perturbation": {name: value.detach() for name, value in initial.items()}, "chosen_perturbation": chosen, "candidate_policy": self.candidate_policy, "initialization": self.initialization, "candidate_probe": self.candidate_probe, "incoming_probe_computed": need_incoming}
        return loss, diagnostics
