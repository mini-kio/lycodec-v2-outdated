# LyCodec v2 — 이전 실험 코드

오디오 인코딩·복원과 스트리밍을 실험한 LyCodec v2 코드입니다. 저장소 이름의 `outdated` 표기처럼 이전 버전 참고용 코드로 분류합니다.

## 구성

| 경로 | 역할 |
| --- | --- |
| [config.yaml](config.yaml) | 오디오·데이터·모델·학습 설정 |
| [train.py](train.py) | Accelerate 기반 학습 진입점 |
| [lycodec/models.py](lycodec/models.py) | 모델 |
| [lycodec/audio.py](lycodec/audio.py) | 오디오 처리 |
| [lycodec/training.py](lycodec/training.py) | 트레이너 |
| [lycodec/inference.py](lycodec/inference.py) | 추론 |
| [lycodec/streaming.py](lycodec/streaming.py) | 스트리밍 |

## 학습 진입점

PyTorch, Accelerate, PyYAML, NumPy, SoundFile 등 코드의 의존성을 준비하고 `config.yaml`의 데이터 경로를 수정한 뒤 실행합니다.

```bash
python train.py --config config.yaml
```

제공된 설정은 44.1 kHz 오디오와 4초 세그먼트를 사용합니다. 설정 파일의 실제 값을 기준으로 실험을 구성하세요.

다른 공개 코덱 실험: [minini](https://github.com/mini-kio/minini), [devcodec](https://github.com/mini-kio/devcodec).
