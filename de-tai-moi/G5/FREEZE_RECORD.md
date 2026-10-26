# Bản ghi đóng băng G5 — 26/10/2026

Bản đóng băng được định danh bằng SHA-256 của từng file tại thời điểm thầy duyệt [GATE_REVIEW_G5](GATE_REVIEW_G5.md), và bằng **commit `c4da1f4`** của kho git (commit đầu tiên, khởi tạo 26/10/2026; mọi file trong bảng dưới khớp mã băm ở commit đó). Kiểm tra lại: `sha256sum <file>` từ thư mục `de-tai-moi/`. Kiểm thay đổi sau đóng băng: `git diff c4da1f4 -- de-tai-moi/G5/EXPERIMENT_PROTOCOL.md` — mọi khác biệt phải nằm trong Mục 11.

Sau khi đóng băng, giao thức chỉ được thay đổi bằng cách thêm dòng vào Mục 11 (Deviations); vì vậy mã băm hiện tại của `EXPERIMENT_PROTOCOL.md` sẽ khác mã băm dưới đây, và mọi khác biệt phải nằm trong Mục 11. File này không tự chứa mã băm của chính nó. `RESEARCH_PLAN.md` và `CLAIM_EVIDENCE_MATRIX.md` là tài liệu sống (thay đổi ghi ở §9 và cột trạng thái); mã băm dưới đây là trạng thái của chúng lúc đóng băng.

## Giao thức và tài liệu đi kèm
| File | Byte | SHA-256 |
|---|---:|---|
| `G5/EXPERIMENT_PROTOCOL.md` | 30.962 | `f2b2a5923928839c4fd9d9a677ae3945a81e69740e3aa892fe39b25c64b796f8` |
| `G5/DATASET_CARDS.md` | 13.594 | `e7f1ef47b503db42bfbfc23b02d84aae78e7d8d1c1b2ae65f9d2fd8a5d1935a5` |
| `G5/GATE_REVIEW_G5.md` | 10.721 | `fc0661aa6ca86121a88a0b8aeca269930084b70cbfdc03b4ad971ba9194c43e5` |
| `G3/RESEARCH_PLAN.md` | 18.281 | `a5165f22ef632f577e22a70d0ae23a7202d3c4863c3e91c19798e9f8cf6890d0` |
| `G3/CLAIM_EVIDENCE_MATRIX.md` | 5.865 | `1a3f53f35d57286b2afa7ebad540c11224f317645203b252bdb25609b535f17d` |

## Code tại thời điểm đóng băng (phương pháp, thí điểm, xác minh dữ liệu)
| File | Byte | SHA-256 |
|---|---:|---|
| `G4/code/icgibbs/__init__.py` | 519 | `cbf85cd7126b79d04cbf8473d19859fddb57b78169359de473751453974ebff0` |
| `G4/code/icgibbs/baselines.py` | 13.750 | `6c289939cf20b29af5943db27e060c9a90063e2eeefa9658ab2fec68c595e2d1` |
| `G4/code/icgibbs/conditional.py` | 10.837 | `4c7edb10f20138f82e76cf69e79f941c7923bc01846b8e490e6aa3945f0d9236` |
| `G4/code/icgibbs/exact.py` | 13.898 | `259e319d8c56620173c2b8e0e58f3850f10e027fa7bbfaf51d9f8f5999e831ba` |
| `G4/code/icgibbs/llp.py` | 4.581 | `ce5ed17bed5009a320d06792968e2ece7228f74976545236383364d4a08e0ee1` |
| `G4/code/icgibbs/sampler.py` | 16.262 | `91314a3734320a3d19a1395d866c5fcf31cf19ff38f22f730389799a9f700a78` |
| `G4/code/icgibbs/weak.py` | 11.138 | `3d9e7659cda31cd264587e67fdb82d54848a5364125c81d5bcc3fbb8b55f2e61` |
| `G4/code/g5_verify_datasets.py` | 3.922 | `da36aa93bb3b71c2a787869d89812fd7fa72565fbdd37dec253f6b0bd3a7e24d` |
| `G4/code/g5_verify_pll_real.py` | 9.482 | `8b6d4e7e05f0689ae098b0ebfeb230a01555348dd353fad6fedb99b16a92922c` |
| `G4/code/kaggle_g4_manyclass_pilot.py` | 16.194 | `8a52c46f8fb5778a5ad38a336b336b94a072ac7a8facd5ddceb6d4ca40c1ae7f` |
| `G4/code/kaggle_g4_pilot.py` | 25.573 | `5eaaa9bc8b148505e2aaa2e58ae20ff5367dc627a7614a9613700744df4f706e` |
| `G4/code/kaggle_g4_rq4_pilot.py` | 17.822 | `a983625788cdc6d950ec06edf62f1aa4b344e43524f328d7cb2d71fbb4b628e2` |
| `G4/code/tests/test_exact.py` | 9.562 | `a00306800de77a8da6602ca8152276144660beaed05de94973fdeab5bec38af6` |
| `G4/code/experiments/cell_pfn.py` | 6.472 | `bb2e385717d65f72da196749e3698a83730217d5fdb65a82686ea717a9cf5bed` |
| `G4/code/experiments/drift_check.py` | 2.007 | `f7c30f09318ef9d06eaa382e6235bcc833c17860af598f03cfdf285f1ed22129` |
| `G4/code/experiments/in_prior_check.py` | 5.145 | `dde4fcd4ef04fe5d5507d3e9dfefb374b5daeaf033f8df199734366135f4f9c3` |
| `G4/code/experiments/rq1_exact.py` | 24.832 | `5a7a2c5f354f787b923e8aa2d0a1115eb348fb64198c6185f5ad99c848848468` |
| `G4/code/experiments/rq1_figures.py` | 4.915 | `b2a5d7957284f76781ff122d9c555c884744878e9e4873dca11213df075d33c5` |
| `G4/code/experiments/warm_start_check.py` | 4.903 | `5cb620627c37398eca22fdb81a2c55b1d5b37ad896185ccbfbe6d6702c870636` |

## Dữ liệu thí điểm và xác minh (chỉ tập phát triển; tập đánh giá chỉ được đọc metadata)
| File | Byte | SHA-256 |
|---|---:|---|
| `G4/pilot/g5_dataset_check.csv` | 3.145 | `c6dc1132e1d2b4917c15ef532d180f690b1c5e389da030abb5aacd251a6d6449` |
| `G4/pilot/g5_pll_real_check.csv` | 1.672 | `0a22b1e13ab24575d9aed9f54cb45fbbb910c0ad709456309627907e9a16e33a` |
| `G4/pilot/g4_pilot_v1.csv` | 81.337 | `c9f43a10993926c7d7718ade1e6c7fe810058f54b7f42eab7feb738496dd57e7` |
| `G4/pilot/g4_pilot_v2.csv` | 146.046 | `c29e4ac3da3754ad814fcd05d47b349f03c9d927fe10929366bd30d6450d6746` |
| `G4/pilot/g4_rq4_pilot.csv` | 82.466 | `a088dc522a05f3fc6b3818bc0fe5e9d1737081a37315f99e9271e8472960e18a` |
| `G4/pilot/g4_manyclass_pilot.csv` | 12.729 | `3fd453e57b3a6627253778aee7bff34d475daa6224ceedc7b4b2278a47c6ef65` |

## Dữ liệu PLL thật (tệp gốc, theo F7)
Lost `ce4c1e0d5ad117e1b48f9f83805cba46f2a955ca5bdf75e578026ae4f6c7a960` · MSRCv2 `8734cbd7b2f96623bfa8a0ea037daa4111fbce3954f077bf6c53d23dfb5e7a3c` · BirdSong `8c0c9b1828e523b6bd499242a94a360a55326c326fe31b788138b3537bc1c33f`
