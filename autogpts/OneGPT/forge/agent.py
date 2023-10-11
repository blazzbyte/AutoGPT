from forge.sdk import (
    Agent,
    AgentDB,
    ForgeLogger,
    Step,
    StepRequestBody,
    Task,
    TaskRequestBody,
    Workspace,    
    PromptEngine,
    Status,	
    chat_completion_request,	
    ChromaMemStore	
)
import json	
import pprint

LOG = ForgeLogger(__name__)

MODEL_NAME = "gpt-3.5-turbo"  #gpt-4


class ForgeAgent(Agent):

    def __init__(self, database: AgentDB, workspace: Workspace):
        super().__init__(database, workspace)

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        task = await super().create_task(task_request)
        LOG.info(
            f"📦 Task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
        )
        return task

    async def plan_steps(self, task, step_request: StepRequestBody):
        step_request.name = "Plan steps"

        if not step_request.input:
            step_request.input = "Create steps to accomplish the objective"

        step = await self.db.create_step(
            task_id=task.task_id, input=step_request, is_last=False
        )

        files = self.workspace.list(task.task_id, "/")

        prompt_engine = PromptEngine("plan-steps")
        task_kwargs = {
            "abilities": self.abilities.list_abilities_for_prompt(),
            "files": files
        }
        system_prompt = prompt_engine.load_prompt("system-prompt",  **task_kwargs)
        system_format = prompt_engine.load_prompt("step-format")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": system_format},
        ]

        task_kwargs = {
            "task": task.input,
        }
        task_prompt = prompt_engine.load_prompt("user-prompt",  **task_kwargs)
        messages.append({"role": "user", "content": task_prompt})

        answer = await self.do_steps_request(messages, new_plan=True)

        await self.create_steps(task.task_id, answer["steps"])
        await self.db.update_step(task.task_id, step.step_id, "completed", output=answer["thoughts"]["text"])

        return step

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        """
        For a tutorial on how to add your own logic please see the offical tutorial series:
        https://aiedge.medium.com/autogpt-forge-e3de53cc58ec

        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to execute
        a step.

        The task that is created contains an input string, for the benchmarks this is the task
        the agent has been asked to solve and additional input, which is a dictionary and
        could contain anything.

        If you want to get the task use:

        ```
        task = await self.db.get_task(task_id)
        ```

        The step request body is essentially the same as the task request and contains an input
        string, for the benchmarks this is the task the agent has been asked to solve and
        additional input, which is a dictionary and could contain anything.

        You need to implement logic that will take in this step input and output the completed step
        as a step object. You can do everything in a single step or you can break it down into
        multiple steps. Returning a request to continue in the step output, the user can then decide
        if they want the agent to continue or not.
        """

        task = await self.db.get_task(task_id)

        steps, page = await self.db.list_steps(task_id, per_page=100)

        if not steps:
            return await self.plan_steps(task, step_request)
        
        
        previous_steps = []
        next_steps = []
        for step in steps:
            if step.status == Status.created:
                next_steps.append(step)
            elif step.status == Status.completed:
                previous_steps.append(step)
        
        if not next_steps:
            LOG.info(f"Tried to execute with no next steps, return last step as the last")
            step = previous_steps[-1]
            step.is_last = True
            return step
        # An example that
        step = await self.db.create_step(
            task_id=task_id, input=step_request, is_last=True
        )

        self.workspace.write(task_id=task_id, path="output.txt", data=b"Washington D.C")

        await self.db.create_artifact(
            task_id=task_id,
            step_id=step.step_id,
            file_name="output.txt",
            relative_path="",
            agent_created=True,
        )

        step.output = "Washington D.C"

        LOG.info(f"\t✅ Final Step completed: {step.step_id}")

        return step
