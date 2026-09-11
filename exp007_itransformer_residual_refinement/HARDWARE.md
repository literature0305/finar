# exp007 — 동작 검증과 하드웨어 요구사항

측정 환경: WSL2 / **RTX 5070 12 GB** / 18 vCPU / 23 GB RAM / torch 2.11.0+cu128,
CPU 상한 8 (`--num-workers 8`), `--loader-workers 0`(기본).

논문은 **NVIDIA P100 16 GB 1장**, batch 32, 10 epochs로 전체 표를 냈다고
명시한다 (arXiv:2310.06625v3, Appendix A "All the experiments … conducted on a
single NVIDIA P100 16GB GPU").

---

## 1. 훈련·평가가 실제로 동작하는가

`train.sh` → `train.py` → (자동) `run_eval.py` 경로를 공식 설정 그대로 돌려
Table 10과 대조했다. 전부 **full run** — 공식 하이퍼파라미터, 10 epoch 상한 +
early stopping, seed 2023.

| 데이터셋 | H | 논문 MSE / MAE | 측정 MSE / MAE | ΔMSE | 판정 |
|---|---|---|---|---|---|
| ETTh1 | 96 | 0.386 / 0.405 | 0.3868 / 0.4052 | +0.2 % | MATCH |
| ETTh1 | 192 | 0.441 / 0.436 | 0.4424 / 0.4362 | +0.3 % | MATCH |
| ETTh1 | 336 | 0.487 / 0.458 | 0.4910 / 0.4622 | +0.8 % | MATCH |
| ETTh1 | 720 | 0.503 / 0.491 | 0.5093 / 0.4938 | +1.3 % | MATCH |
| ETTh2 | 96 | 0.297 / 0.349 | 0.2994 / 0.3490 | +0.8 % | MATCH |
| ETTh2 | 192 | 0.380 / 0.400 | 0.3822 / 0.3987 | +0.6 % | MATCH |
| ETTh2 | 336 | 0.428 / 0.432 | 0.4209 / 0.4312 | −1.6 % | MATCH |
| ETTh2 | 720 | 0.427 / 0.445 | 0.4310 / 0.4467 | +0.9 % | MATCH |
| ETTm1 | 96 | 0.334 / 0.368 | 0.3430 / 0.3772 | +2.7 % | MATCH |
| ETTm2 | 96 | 0.180 / 0.264 | 0.1856 / 0.2723 | +3.1 % | MATCH |
| Weather | 96 | 0.174 / 0.214 | 0.1778 / 0.2179 | +2.2 % | MATCH |

11셀 전부 허용 오차(상대 5 %, 절대 0.005 — 논문 Table 5의 seed 표준편차에서
산출, `paper.py` 참조) 안이다. 최대 편차는 ETTm2/96의 +3.1 %.

수치 말고 같이 확인한 것:

* `--refinement off` / `on` 두 시나리오가 같은 플래그 집합으로 돌아간다.
  baseline에서는 체인 설정이 무시되며 그 사실이 로그에 남는다.
* 훈련이 끝나면 `run_eval.py`가 자동으로 돌아 `metrics.json`을 남긴다.
* `eval.sh --depths "1 2 3 4"`로 재훈련 없이 깊이별 재채점이 된다
  (한 번의 forward에서 모든 깊이를 읽는 것은 precheck R5가 보증).
* `build_table.py`가 protocol이 일치하는 baseline/refinement만 짝짓는다.
* `train.sh --mode job --dry-run`이 컨테이너에서 돌 ssub 명령을 만든다.
* `precheck.py` 6개 체크 전부 PASS — 공식 repo와 bit-identical forecast,
  9개 데이터셋 전 split window 단위 일치 포함.

---

## 2. 셀별 실측 자원

`GPU peak`는 torch allocator의 live tensor 최대치, `GPU reserved`는 드라이버
에서 실제 확보한 양 — **VRAM에 들어가야 하는 값은 reserved**다. `host`는
프로세스 RSS 최대치(코퍼스 파싱 포함), `s/epoch`는 훈련 + 검증 1 epoch
(테스트 채점은 별도).

| 데이터셋 | 변량 | H | d_model | e_l | batch | 파라미터 | GPU peak | GPU reserved | host | s/epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Exchange | 8 | 96 | 128 | 2 | 32 | 0.22 M | 0.03 | 0.03 | 1.45 | 1.2 |
| Exchange | 8 | 720 | 128 | 2 | 32 | 0.30 M | 0.03 | 0.04 | 1.45 | 1.1 |
| ETTh1 | 7 | 96 | 256 | 2 | 32 | 0.84 M | 0.04 | 0.06 | 1.45 | 1.5 |
| ETTh1 | 720 | | 512 | 2 | 32 | 3.58 M | 0.09 | 0.11 | 1.47 | 1.6 |
| ETTh2 | 7 | 96–720 | 128 | 2 | 32 | 0.22–0.30 M | 0.03 | 0.03–0.04 | 1.45 | 1.5–1.7 |
| ETTm1/m2 | 7 | 96–720 | 128 | 2 | 32 | 0.22–0.30 M | 0.03 | 0.03 | 1.45 | 6.3–6.7 |
| Weather | 21 | 96 | 512 | 3 | 32 | 4.83 M | 0.14 | 0.17 | 1.47 | 10.0 |
| Weather | 21 | 720 | 512 | 3 | 32 | 5.15 M | 0.15 | 0.19 | 1.48 | 9.1 |
| Solar | 137 | 96 | 512 | 2 | 32 | 3.26 M | 0.38 | 0.46 | 1.49 | 14.0 |
| Solar | 137 | 720 | 512 | 2 | 32 | 3.58 M | 0.43 | 0.55 | 1.54 | 16.6 |
| ECL | 321 | 96 | 512 | 3 | 16 | 4.83 M | 0.79 | 0.92 | 1.52 | 30.3 |
| ECL | 321 | 720 | 512 | 3 | 16 | 5.15 M | 0.87 | 1.03 | 1.58 | 31.1 |
| **Traffic** | **862** | 96 | 512 | 4 | 16 | 6.41 M | **4.95** | **5.71** | 1.56 | **93.8** |
| **Traffic** | **862** | 720 | 512 | 4 | 16 | 6.73 M | **5.02** | **5.75** | 1.68 | **99.0** |

(GB / 초. Weather·Solar·Exchange·ECL·Traffic 행은 자원 측정용 1-epoch 실행 —
allocator 최대치는 첫 epoch에 도달하므로 VRAM 값은 full run과 같고, s/epoch도
full run에서 나온 값과 일치한다.)

**메모리를 결정하는 것은 변량 수다.** iTransformer는 변량 하나를 토큰 하나로
쓰므로 attention이 O(N²)이고, N=862(Traffic)는 N=321(ECL)의 5.5배, N=21
(Weather)의 30배를 쓴다. horizon은 거의 무관하다 — Traffic 96 → 720에서
파라미터만 0.3 M 늘고 활성값은 사실상 그대로다(5.71 → 5.75 GB).

---

## 3. 하드웨어 요구사항

| 항목 | 최소 | 권장 | 근거 |
|---|---|---|---|
| **GPU VRAM** | **8 GB** | **16 GB** | 최대 셀(Traffic/720) reserved 5.75 GB. 8 GB면 들어가되 여유가 없고, 16 GB면 논문과 동일(P100 16 GB)하며 refinement·batch 증설까지 수용 |
| GPU 대수 | 1 | 1 | 가장 큰 셀이 6 GB 미만이라 분산의 이득이 없다. 논문도 P100 1장 |
| 호스트 RAM | **4 GB** | 8 GB | 실측 최대 RSS **1.68 GB** (Traffic 136 MB csv 파싱 포함). 코퍼스를 한 번만 읽으므로 데이터셋 크기에 거의 무관 |
| CPU | 2 코어 | 8 코어 | `--loader-workers 0`이 기본(실측상 가장 빠름)이라 메인 프로세스의 BLAS만 쓴다. `--num-workers`로 건 상한 안에 머문다 |
| 디스크 | 1 GB | 5 GB | 데이터 420 MB + 공식 checkout 30 MB + 체크포인트(36셀 × 1–27 MB) |
| 네트워크 | 최초 1회 | — | `prepare_data.sh`가 HF/GitHub에서 420 MB를 받는다. 이후 오프라인 가능 |

GPU 없이 CPU만으로도 ETT 계열은 돌지만(`--device cpu`), Traffic은 현실적이지
않다. **A100 40 GB 1장이면 전체 표에 충분하고도 남는다** — 오히려 서로 다른
데이터셋 6–8개를 한 카드에 동시에 올릴 수 있다.

---

## 4. 전체 표(9 데이터셋 × 4 horizon = 36셀) 소요 시간

RTX 5070 실측 s/epoch 기준. 실측 early stopping은 4–8 epoch에서 멈췄다(평균
약 6), 그래서 "실측 기준"은 상한의 0.6배로 잡았다.

| 데이터셋 | s/epoch | 10 epoch 상한 (4 horizon) | 실측 기준 |
|---|---:|---:|---:|
| ETTh1 | 1.6 | 64 s | ~38 s |
| ETTh2 | 1.6 | 64 s | ~38 s |
| ETTm1 | 6.5 | 260 s | ~156 s |
| ETTm2 | 6.5 | 260 s | ~156 s |
| Exchange | 1.2 | 48 s | ~29 s |
| Weather | 9.6 | 384 s | ~230 s |
| Solar | 15.3 | 612 s | ~367 s |
| ECL | 30.7 | 1 228 s | ~737 s |
| Traffic | 96.4 | 3 856 s | ~2 314 s |
| **합계** | | **6 776 s ≈ 1.9 시간** | **≈ 1.1 시간** |

Traffic 한 데이터셋이 전체의 57 %를 차지한다. A100은 이 카드보다 대략
1.5–2배 빠르므로 **전체 표가 A100 1장에서 1시간 내외**로 끝난다. 테스트 채점
시간은 여기 포함되지 않았다(셀당 수 초 ~ 2분, Traffic이 가장 길다).

---

## 5. residual refinement의 추가 비용

체인 깊이 K는 encoder를 K번 돌린다. 기본 `--coe-backprop last`에서는 pass
1..N−1이 `no_grad`라 활성값이 쌓이지 않아 **메모리는 사실상 그대로이고 시간만
늘어난다**. 훈련은 매 스텝 N ~ U{1..K}를 뽑으므로 평균 (K+1)/2 패스다.

| 셀 | baseline reserved / s·ep | `--train-depth 3` reserved / s·ep | 배수 |
|---|---|---|---|
| ETTh1 / 96 | 0.06 GB / 1.5 s | 0.06 GB / 2.4 s | 메모리 ×1.0, 시간 ×1.6 |
| Weather / 96 | 0.17 GB / 10.0 s | 0.18 GB / 10.8 s | 메모리 ×1.06, 시간 ×1.08 |
| ECL / 96 | 0.92 GB / 30.3 s | 0.93 GB / 39.1 s | 메모리 ×1.01, 시간 ×1.29 |

`--coe-backprop all`(BPTT 전개)은 이야기가 다르다. ECL/96에서 K=3 고정
(`--coe-stochastic-repeat false`)으로 직접 비교한 결과:

| backprop | GPU reserved | s/epoch |
|---|---|---|
| `last` (기본) | 0.88 GB | 48.5 s |
| `all` | **2.35 GB (×2.7)** | 77.1 s (×1.6) |

K=3에서 메모리가 2.7배 — K배에 못 미치는 것은 파라미터와 optimizer state가
패스 수에 비례하지 않기 때문이다. 이 배수를 최대 셀에 적용하면 Traffic/720이
5.75 → **약 15 GB**가 되어 16 GB 카드로는 아슬아슬하다(이 수치는 ECL에서 잰
배수를 옮긴 추정이며, Traffic에서 직접 재지는 않았다). `--coe-backprop all`을
쓸 계획이라면 **A100 40 GB를 권한다**. 기본값 `last`를 쓰는 한 baseline과 같은
카드로 충분하다.

임베딩이 `seq_len` → `seq_len + pred_len`으로 넓어져 파라미터는
`d_model × pred_len`만큼 는다(Traffic/720에서 +0.37 M, 약 6 %).

---

## 6. 원격 A100에서 돌리는 법

```bash
# 1) 한 번만: 데이터 + 공식 checkout (420 MB)
bash prepare_data.sh --data-root /group-volume/ts-dataset/ltsf --with-reference

# 2) 전체 표, baseline — 한 job이 36셀을 순차 실행하고 각각 채점까지 한다
bash train.sh --mode job --ngpu 1 \
     --dataset "ETTh1 ETTh2 ETTm1 ETTm2 ECL Traffic Weather Exchange Solar" \
     --pred-len "96 192 336 720" --refinement off

# 3) 같은 설정으로 refinement
bash train.sh --mode job --ngpu 1 \
     --dataset "ETTh1 ETTh2 ETTm1 ETTm2 ECL Traffic Weather Exchange Solar" \
     --pred-len "96 192 336 720" --refinement on \
     --train-depth 3 --depth 3 --eval-depths "1 2 3 4"

# 4) 표와 그림
bash eval.sh --stage table
```

Traffic만 떼어 병렬로 돌리려면 job을 둘로 나누면 된다 — 결과 디렉터리 이름이
설정 digest로 갈리므로 같은 `--out`을 써도 섞이지 않고, `build_table.py`는
protocol이 맞는 짝만 비교한다.

`--num-workers 8|16|32`로 CPU 상한을 명시할 수 있다. 지정하지 않으면 cgroup
할당량과 affinity mask에서 스스로 읽는다.

각 실행은 `train_log.json`에 `resources`(GPU/host 최대치, epoch 시간,
파라미터 수)를 남기므로, 다른 카드에서의 실제 수치는 그 파일에서 바로 읽을 수
있다.
