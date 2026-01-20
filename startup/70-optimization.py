
from blop import RangeDOF, Objective, Agent

from startup.optimization.alignment import ElectrometerEvaluation



# QuadEM alignment optimization with M1 Mirror and Hexapod Mirrors
sensors = [xqem01]
dofs = [
    #RangeDOF(
    #    actuator=M1.X,
    #    bounds=(-4.0, -2.6),
    #    #bounds=(-4.3, -4.2),
    #    #step_size=0.1,
    #    parameter_type="float",
    #),
    #RangeDOF(
    #    actuator=M1.Ry,
    #    bounds=(-3744.0, -3604.0),
    #    #bounds=(-4674.0, -4673.0),
    #    #step_size=1.0,
    #    parameter_type="float",
    #),
    # RangeDOF(
    #     actuator=M3.X,
    #     bounds=(-0.5, 0.5),
    #     #bounds=(0.4, 0.5),
    #     #step_size=0.01,
    #     parameter_type="float",
    # ),
    #RangeDOF(
    #    actuator=M3.Y,
    #    bounds=(13.5, 14.5),
    #    #bounds=(14.0, 14.1),
    #    #step_size=0.1,
    #    parameter_type="float",
    #),
    RangeDOF(
        actuator=M3.Z,
        bounds=(-0.5, 0.5),
        #bounds=(0.0, 0.1),
        #step_size=0.1,
        parameter_type="float",
    ),
    #RangeDOF(
    #    actuator=M3.Rx,
    #    bounds=(-0.5, 0.5),
    #    #bounds=(0.0, 0.1),
    #    #step_size=0.1,
    #    parameter_type="float",
    #),
    RangeDOF(
        actuator=M3.Ry,
        bounds=(-0.712, -0.709),
        #bounds=(-0.7, -0.69),
        #step_size=0.00005,
        parameter_type="float",
    ),
    #RangeDOF(
    #    actuator=M3.Rz,
    #    bounds=(-0.5, 0.5),
    #    #bounds=(0.0, 0.1),
    #    #step_size=0.01,
    #    parameter_type="float",
    #),
]
objectives = [
    Objective(name="xqem01_current", minimize=False),
]

# Simple EM alignment agent that uses the "fast" generation strategy
# starting from randomly sampled points
em_align_agent1 = Agent(
    sensors=sensors,
    dofs=dofs,
    objectives=objectives,
    evaluation_function=ElectrometerEvaluation(tiled_client=tiled_reading_client),
    checkpoint_path="/nsls2/data/esm/legacy/optimization/alignment/em_alignment.json",
    name="simple_em_alignment",
)
em_align_agent1.ax_client.configure_generation_strategy(
    method="fast",
    initialize_with_center=True,
    allow_exceeding_initialization_budget=True,
)

# Simple EM alignment agent that uses the "quality" generation strategy
# starting from randomly sampled points
em_align_agent2 = Agent(
    sensors=sensors,
    dofs=dofs,
    objectives=objectives,
    evaluation_function=ElectrometerEvaluation(tiled_client=tiled_reading_client),
    checkpoint_path="/nsls2/data/esm/legacy/optimization/alignment/em_alignment.json",
    name="simple_em_alignment",
)
em_align_agent2.ax_client.configure_generation_strategy(
    method="fast",
    initialization_budget=25,
    initialize_with_center=True,
    allow_exceeding_initialization_budget=True,
)

