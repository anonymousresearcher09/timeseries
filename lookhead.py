""" Lookahead Optimizer Wrapper.
Implementation modified from: https://github.com/alphadl/lookahead.pytorch
Paper: `Lookahead Optimizer: k steps forward, 1 step back` - https://arxiv.org/abs/1907.08610
Hacked together by / Copyright 2020 Ross Wightman
"""
from collections import defaultdict
import torch
from torch.optim import Optimizer

class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')

        self.base_optimizer = base_optimizer
        # Lookahead용 기본값(파라미터 그룹에 주입)
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)

        # 🔴 핵심: Optimizer 내부 훅/상태 초기화를 위해 반드시 호출
        # base_optimizer.param_groups 를 그대로 넘겨 동일한 그룹을 공유합니다.
        super().__init__(self.base_optimizer.param_groups, defaults)

        self.state = defaultdict(dict)

        # 파라미터 그룹에 기본값 보장
        for group in self.param_groups:
            group.setdefault('lookahead_alpha', alpha)
            group.setdefault('lookahead_k', k)
            group.setdefault('lookahead_step', 0)

    def update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if 'slow_buffer' not in param_state:
                param_state['slow_buffer'] = torch.empty_like(fast_p.data)
                param_state['slow_buffer'].copy_(fast_p.data)
            slow = param_state['slow_buffer']
            slow.add_(group['lookahead_alpha'], fast_p.data - slow)
            fast_p.data.copy_(slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            group['lookahead_step'] += 1
            if group['lookahead_step'] % group['lookahead_k'] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.base_optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = {
            'state': state_dict['state'],
            'param_groups': state_dict['param_groups'],
        }
        self.base_optimizer.load_state_dict(fast_state_dict)

        # slow buffer 복구
        if 'slow_state' not in state_dict:
            print('Loading state_dict from optimizer without Lookahead applied.')
            state_dict['slow_state'] = defaultdict(dict)

        slow_state_dict = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],
        }
        # Optimizer.load_state_dict 사용하려면 super().__init__로 초기화되어 있어야 함
        super(Lookahead, self).load_state_dict(slow_state_dict)

        # 동일 컨테이너 참조 유지
        self.param_groups = self.base_optimizer.param_groups

        # 누락된 lookahead 기본값 주입(재보장)
        for group in self.param_groups:
            group.setdefault('lookahead_alpha', self.defaults['lookahead_alpha'])
            group.setdefault('lookahead_k', self.defaults['lookahead_k'])
            group.setdefault('lookahead_step', 0)
