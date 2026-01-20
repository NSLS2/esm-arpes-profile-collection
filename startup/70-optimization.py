
from blop import RangeDOF, Objective, Agent

from .optimization.alignment import ElectrometerEvaluation



# QuadEM alignment optimization with M1 Mirror and Hexapod Mirrors
sensors = [xqem01.current1]
dofs = [
    RangeDOF(
        actuator=M1_mirror.X,
        bounds=(-4.35, -2.35),
        step_size=0.05,
        parameter_type="float",
    ),
    RangeDOF(
        actuator=M1_mirror.Ry,
        bounds=(-4674, -2674),
        step_size=1.0,
        parameter_type="float",
    ),
    RangeDOF(
        actuator=Hexapod_Mir.X,
        bounds=(-0.5, 0.5),
        step_size=0.01,
        parameter_type="float",
    ),
    RangeDOF(
        actuator=Hexapod_Mir.Y,
        bounds=(13.0, 15.0),
        step_size=0.1,
        parameter_type="float",
    ),
    RangeDOF(
        actuator=Hexapod_Mir.Z,
        bounds=(-1.0, 1.0),
        step_size=0.1,
        parameter_type="float",
    ),
    RangeDOF(
        actuator=Hexapod_Mir.Rx,
        bounds=(-1.0, 1.0),
        step_size=0.1,
        parameter_type="float",
    ),
    RangeDOF(
        actuator=Hexapod_Mir.Ry,
        bounds=(-0.76, -0.66),
        step_size=0.00005,
        parameter_type="float",
    ),
    RangeDOF(
        actuator=Hexapod_Mir.Rz,
        bounds=(-1.0, 1.0),
        step_size=0.01,
        parameter_type="float",
    ),
]
objectives = [
    Objective("xqem01_current", minimize=False),
]

# Simple EM alignment agent that uses the "fast" generation strategy
# starting from randomly sampled points
em_align_agent1 = Agent(
    sensors=sensors,
    dofs=dofs,
    objectives=objectives,
    evaluation_function=ElectrometerEvaluation(tiled_client=tiled_reading_client),
    checkpoint_path="/nsls2/data/esm/legacy/optimization/alignment/em_alignment.json",
    client_kwargs={"name": "simple_em_alignment"},
)
em_align_agent1.ax_client.configure_generation_strategy(
    method="fast",
    initialize_with_center=False,
    allow_exceeding_initialization_budget=False,
)

# Simple EM alignment agent that uses the "quality" generation strategy
# starting from randomly sampled points
em_align_agent2 = Agent(
    sensors=sensors,
    dofs=dofs,
    objectives=objectives,
    evaluation_function=ElectrometerEvaluation(tiled_client=tiled_reading_client),
    checkpoint_path="/nsls2/data/esm/legacy/optimization/alignment/em_alignment.json",
    client_kwargs={"name": "simple_em_alignment"},
)
em_align_agent2.ax_client.configure_generation_strategy(
    method="quality",
    use_existing_trials_for_initialization=True,
)
