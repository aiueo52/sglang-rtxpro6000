# SPDX-License-Identifier: Apache-2.0
"""Resolve launches without starting a server, touching models, or using CUDA."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    'SGLANG_MOE_PRUNE_IN_PROLOGUE': '0', 'SGLANG_EXPERT_TIER_K': '320',
    'SGLANG_EXPERT_TIER_SOURCE': 'mmap', 'SGLANG_EXPERT_TIER_GATHER': 'v2',
    'SGLANG_EXPERT_TIER_GATHER_CHUNK': '8192', 'SGLANG_EXPERT_TIER_GATHER_CTAS': '376',
    'SGLANG_EXPERT_TIER_PRETOUCH': '1', 'SGLANG_EXPERT_TIER_DECIDE': 'sort',
    'SGLANG_EXPERT_TIER_PREDICT': '0', 'SGLANG_EXPERT_TIER_KEEP': '60',
    'SGLANG_EXPERT_TIER_TIMING': '0', 'SGLANG_EXPERT_TIER_ARENA': 'v2',
    'SGLANG_EXPERT_TIER_PREFETCH': '0', 'SGLANG_EXPERT_TIER_POLICY': 'lru',
    'SGLANG_EXPERT_TIER_SKIP_EMPTY': '0', 'SGLANG_EXPERT_TIER_FUSED': '0', 'SGLANG_RND_PLE_META': '1',
}


def launch(**overrides):
    env = {'PATH': os.environ['PATH'], 'HOME': str(ROOT), 'PYTHON': sys.executable,
           'TARGET_MODEL': str(ROOT), 'TOKEN_MAP': 'none', 'CONC': '8',
           'CUDA_VISIBLE_DEVICES': '', 'FLASH_NEXT_ADMIN_API_KEY': 'cpu-test-placeholder'}
    env.update(overrides)
    return subprocess.run(['bash', str(ROOT / 'serve-fast.sh'), 'wa', '--print'],
                          env=env, capture_output=True, text=True)


@pytest.fixture
def hot(tmp_path):
    path = tmp_path / 'hot set.json'
    path.write_text(json.dumps({'K': 512, 'layers': [list(range(512)) for _ in range(48)]}))
    return str(path)


def test_disabled_preserves_launch():
    baseline = launch()
    disabled = launch(EXPERT_TIER_K='0', EXPERT_TIER_HOT='missing', EXPERT_TIER_KEEP='invalid')
    assert baseline.returncode == disabled.returncode == 0
    assert json.loads(baseline.stdout) == json.loads(disabled.stdout)
    env = json.loads(disabled.stdout)['env']
    assert not any(k.startswith('SGLANG_EXPERT_TIER_') for k in env)
    assert 'SGLANG_RND_PLE_META' not in env
    assert env['SGLANG_MOE_PRUNE_IN_PROLOGUE'] == '1'  # existing launcher default


def test_disabled_preserves_inherited_env():
    result = launch(SGLANG_EXPERT_TIER_K='9', SGLANG_RND_PLE_META='0')
    assert result.returncode == 0
    assert json.loads(result.stdout)['env']['SGLANG_EXPERT_TIER_K'] == '9'
    assert json.loads(result.stdout)['env']['SGLANG_RND_PLE_META'] == '0'


def test_enabled_exports_and_args(hot):
    result = launch(EXPERT_TIER_K='320', EXPERT_TIER_HOT=hot,
                    SGLANG_EXPERT_TIER_SKIP_EMPTY='1', SGLANG_EXPERT_TIER_FUSED='1')
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert {k: data['env'][k] for k in EXPECTED} == EXPECTED
    assert data['env']['SGLANG_EXPERT_TIER_HOT'] == hot
    assert 'required K=' not in result.stderr
    assert '--speculative-adaptive' in data['argv']
    assert data['argv'][len(data['argv']) - data['argv'][::-1].index('--max-running-requests')] == '8'


def test_small_k_warns_but_continues(hot):
    result = launch(EXPERT_TIER_K='319', EXPERT_TIER_HOT=hot, EXPERT_TIER_KEEP='120')
    assert result.returncode == 0
    assert 'required K=320 (M=32' in result.stderr
    assert json.loads(result.stdout)['env']['SGLANG_EXPERT_TIER_KEEP'] == '120'


def test_effective_bucket_range_and_custom_table(hot, tmp_path):
    table = tmp_path / 'table.json'
    table.write_text(json.dumps({'1': {'candidate_steps': [3, 15]},
                                 '5': {'candidate_steps': [3]}}))
    # Bucket 1 covers batches 1..4: max M is 4*16=64, not 1*16 or 5*4.
    result = launch(EXPERT_TIER_K='320', EXPERT_TIER_HOT=hot, ADAPTIVE_CONFIG=str(table))
    assert result.returncode == 0
    assert 'required K=640 (M=64' in result.stderr


@pytest.mark.parametrize('concurrency,required', [('1', 160), ('4', 320), ('6', 480), ('8', 320)])
def test_selected_table(hot, concurrency, required):
    result = launch(EXPERT_TIER_K='1', EXPERT_TIER_HOT=hot, CONC=concurrency)
    assert result.returncode == 0
    assert f'required K={required} ' in result.stderr


def test_missing_hot_stops(tmp_path):
    result = launch(EXPERT_TIER_K='320', EXPERT_TIER_HOT=str(tmp_path / 'missing.json'))
    assert result.returncode != 0
    assert 'HOT file not found' in result.stderr and not result.stdout


def test_default_hot_path():
    # Do not create a deployment hot set just to make this test pass.
    if (ROOT / 'expert-tier/hot-set.json').exists():
        pytest.skip('deployment hot set exists')
    result = launch(EXPERT_TIER_K='320')
    assert result.returncode != 0
    assert str(ROOT / 'expert-tier/hot-set.json') in result.stderr


@pytest.mark.parametrize('key,value', [('EXPERT_TIER_K', '-1'), ('EXPERT_TIER_K', '513'),
                                       ('EXPERT_TIER_KEEP', '4')])
def test_invalid_settings_stop(hot, key, value):
    settings = {'EXPERT_TIER_K': '320', 'EXPERT_TIER_HOT': hot, key: value}
    result = launch(**settings)
    assert result.returncode != 0 and 'must be' in result.stderr


@pytest.mark.parametrize('settings,expected', [
    ({}, '262144'),
    ({'MAX_TOTAL_TOKENS': ''}, '262144'),
    ({'MAX_TOTAL_TOKENS': '300000'}, '300000'),
    ({'CONTEXT_LENGTH': '524288'}, '524288'),
])
def test_enabled_max_total_tokens(hot, settings, expected):
    result = launch(EXPERT_TIER_K='320', EXPERT_TIER_HOT=hot, **settings)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    index = data['argv'].index('--max-total-tokens')
    assert data['argv'][index + 1] == expected
    assert data['argv'][data['argv'].index('--context-length') + 1] == settings.get(
        'CONTEXT_LENGTH', '262144')


def test_disabled_does_not_default_max_total_tokens():
    result = launch(EXPERT_TIER_K='0', CONTEXT_LENGTH='524288')
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert 'MAX_TOTAL_TOKENS' not in data['env']
    assert '--max-total-tokens' not in data['argv']
