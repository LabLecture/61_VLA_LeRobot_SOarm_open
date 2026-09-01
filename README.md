# SO-101 환경 확인 및 캘리브레이션

이 브랜치는 SO-101 Leader/Follower를 LeRobot에 연결하기 위한 초기 환경 확인과 캘리브레이션까지만 제공합니다.

포함 범위:

- Intel XPU, NVIDIA CUDA 또는 명시적인 CPU PyTorch 환경 확인
- SO-101 Leader/Follower 포트 확인
- Leader/Follower 개별 캘리브레이션

데이터 수집, 웹 GUI, 정책 학습·추론 및 Agent 기능은 포함하지 않습니다. 해당 기능은 필요할 때 이 공개 브랜치에 기능 단위로 별도 반영합니다.

## 준비

프로젝트 가상환경을 활성화합니다.

```powershell
.\.venv\Scripts\Activate.ps1
```

LeRobot과 사용 중인 GPU에 맞는 PyTorch가 설치되어 있어야 합니다. 현재 Python과 LeRobot, PyTorch 및 GPU 연산을 확인하려면 다음 명령을 실행합니다.

```powershell
python .\hello_lerobot.py --device auto
```

`auto`는 CUDA를 먼저 확인하고, 없으면 XPU를 선택합니다. GPU가 없을 때 CPU로 자동 전환하지 않습니다. 장치를 직접 지정할 수도 있습니다.

```powershell
python .\hello_lerobot.py --device xpu
python .\hello_lerobot.py --device cuda
python .\hello_lerobot.py --device cpu
```

## 포트 확인

Leader와 Follower를 한 번에 하나씩 연결해 각 포트를 확인합니다.

```powershell
lerobot-find-port
```

이 프로젝트에서 확인했던 포트는 Follower `COM3`, Leader `COM5`였지만 USB 재연결 후 달라질 수 있습니다. 아래 명령의 포트는 현재 PC에서 확인된 값으로 바꾸십시오.

다른 LeRobot 프로세스, 시리얼 모니터 또는 IDE 확장 기능이 같은 포트를 점유하지 않도록 모두 종료합니다.

## Leader 캘리브레이션

```powershell
lerobot-calibrate `
    --teleop.type=so101_leader `
    --teleop.port=COM5 `
    --teleop.id=so101_leader_main
```

안내가 나오면 다음 순서로 진행합니다.

1. Leader를 관절 가동 범위의 중앙 자세로 옮긴 뒤 Enter를 누릅니다.
2. `wrist_roll`을 제외한 관절을 하나씩 전체 기계 범위로 천천히 움직입니다.
3. MIN/MAX 값이 실제 움직임에 따라 바뀌는지 확인합니다.
4. 모든 관절 범위를 기록한 뒤 Enter를 눌러 저장합니다.

## Follower 캘리브레이션

Follower 작업 공간을 비우고 팔을 손으로 받친 상태에서 실행합니다.

```powershell
lerobot-calibrate `
    --robot.type=so101_follower `
    --robot.port=COM3 `
    --robot.id=so101_follower_main
```

Leader와 같은 방식으로 중앙 자세와 전체 가동 범위를 기록합니다. 캘리브레이션 중에는 모터 전원이나 USB를 분리하지 마십시오.

## 확인 기준과 문제 해결

- 각 암에서 모터 ID 1~6이 모두 검색되어야 합니다.
- 관절을 움직일 때 POS와 MIN/MAX가 함께 변해야 합니다.
- 측정 관절의 MIN/MAX 폭은 최소 100 tick 이상을 권장합니다.
- `There is no status packet`이 나오면 포트 중복 점유, 전원, USB 케이블, 모터 ID와 통신 상태를 먼저 확인합니다.
- `Could not connect on port`가 나오면 `lerobot-find-port`로 포트를 다시 확인합니다.
- 저장된 캘리브레이션과 모터 값이 다르다는 메시지가 나오면 해당 암만 다시 캘리브레이션합니다.

캘리브레이션이 완료된 뒤 데이터 수집을 시작할 때는 동일한 LeRobot ID를 유지해야 합니다.
