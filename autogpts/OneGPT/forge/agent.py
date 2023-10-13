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

from typing import Optional

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
        print(task)
        await self.generate_steps(task)
        return task

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        task = await self.db.get_task(task_id)
        ##Se recupera los pasos creados en el generate_steps
        steps, page = await self.db.list_steps(task_id, per_page=100)
        print(steps)
        step = steps[-1]
        print(step)

        output = await self.abilities.run_ability(
            task_id, "read_file", file_path='file_to_read.txt'
        )
        print(output)
        return step

    async def generate_steps(self, task: Task) -> None:

        prompt_engine = PromptEngine("plan-steps")

        ##files = self.workspace.list(task.task_id, "/")
        abilities = self.abilities.list_abilities_for_prompt()

        ##System Prompt
        task_kwargs = {
            "abilities": abilities,
            ##"files": files
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

        ## Call to Completation Open AI
        chat_completation_kwargs = {
            "messages": messages,
            "model": MODEL_NAME,
        }

        try:
          chat_response = await chat_completion_request(**chat_completation_kwargs)
          print(chat_response)
          response = chat_response["choices"][0]["message"]["content"]
          stepsJSON = json.loads(response)
          for i, step_data in enumerate(stepsJSON["steps"], start=1):
            # Formatear el paso para que se ajuste a StepRequestBody
            step_request = StepRequestBody(
                name=step_data["name"],
                input=step_data["input"],
                additional_input=step_data["additional_input"],
            )

            # Marcar el último paso como is_last
            is_last = i == len(stepsJSON["steps"])

            # Crear el paso en la base de datos
            await self.db.create_step(task_id= task.task_id, input=step_request, additional_input = step_request.additional_input, is_last=is_last)

            LOG.info(f"Create step {i}: {step_data['name']}")
        except Exception as e:
          print('An exception occurred', e)
    
    async def get_current_step(self, task_id: str) -> Optional[Step]:
        """
        Get the currernt step for executing.     
        """
        page = 1  # Página inicial
        per_page = 10  # Cantidad de elementos por página

        while True:
            # Obtener la página actual de pasos
            steps, _ = await self.db.list_steps(task_id=task_id, page=page, per_page=per_page)

            for step in steps:
                if step.status == Status.created.value:
                    return step

            # Si no se encontró un paso en la página actual, pasar a la siguiente página
            if not steps or len(steps) < per_page:
                break  # No hay más páginas

            page += 1

        return None  # No se encontró ningún paso en estado "created"
