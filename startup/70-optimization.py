
from blop import RangeDOF, Objective, Agent

from .optimization.alignment import ElectrometerEvaluation



# QuadEM alignment optimization with M1 Mirror and Hexapod Mirrors
sensors = [qem01.current1]
dofs = [
    RangeDOF(
        actuator=M1_mirror.X,
        bounds=(0, 0), # TODO: set the range
        step_size=None, # TODO: Any step size?
        parameter_type="float",
    ),
    RangeDOF(
        actuator=M1_mirror.Ry,
        bounds=(0, 0), # TODO: set the range
        step_size=None, # TODO: Any step size?
        parameter_type="float",
    ),
    RangeDOF(
        actuator=Hexapod_Mir.X,
        bounds=(0, 0), # TODO: set the range
        step_size=None, # TODO: Any step size?
        parameter_type="float",
    ),
    RangeDOF(
        actuator=Hexapod_Mir.Y,
        bounds=(0, 0), # TODO: set the range
        step_size=None, # TODO: Any step size?
        parameter_type="float",
    ),
    RangeDOF(
        actuator=Hexapod_Mir.Z,
        bounds=(0, 0), # TODO: set the range
        step_size=None, # TODO: Any step size?
        parameter_type="float",
    ),
    RangeDOF(
        actuator=Hexapod_Mir.Rx,
        bounds=(0, 0), # TODO: set the range
        step_size=None, # TODO: Any step size?
        parameter_type="float",
    ),
    RangeDOF(
        actuator=Hexapod_Mir.Ry,
        bounds=(0, 0), # TODO: set the range
        step_size=None, # TODO: Any step size?
        parameter_type="float",
    ),
    RangeDOF(
        actuator=Hexapod_Mir.Rz,
        bounds=(0, 0), # TODO: set the range
        step_size=None, # TODO: Any step size?
        parameter_type="float",
    ),
]
objectives = [
    Objective("electrometer_current", minimize=False),
]

# Simple EM alignment agent that uses the "quality" generation strategy
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
    method="quality",
    initialize_with_center=False,
    allow_exceeding_initialization_budget=False,
)

# TODO: EM alignment agent that will re-sampled points from previous runs (if available)
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

# TODO: EM alignment agent that uses transfer learning from previous runs (if available)
em_align_agent3 = Agent(
    sensors=sensors,
    dofs=dofs,
    objectives=objectives,
    evaluation_function=ElectrometerEvaluation(tiled_client=tiled_reading_client),
    checkpoint_path="/nsls2/data/esm/legacy/optimization/alignment/em_alignment.json",
    client_kwargs={"name": "simple_em_alignment"},
)
em_align_agent3.ax_client.configure_generation_strategy(
    method="quality",
    use_existing_trials_for_initialization=True,
)
