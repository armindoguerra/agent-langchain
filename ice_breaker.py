from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

information = """"
Elon Reeve Musk FRS (Pretória, 28 de junho de 1971) é um empreendedor,[2] empresário e filantropo sul-africano-canadense, naturalizado norte-americano. Ele é o fundador, diretor executivo e diretor técnico da SpaceX; CEO da Tesla, Inc.; vice-presidente da OpenAI, fundador e CEO da Neuralink; cofundador, presidente da SolarCity e proprietário do Twitter. Em dezembro de 2022, tinha uma fortuna avaliada em US$ 139 bilhões de dólares, tornou-se a segunda pessoa mais rica do mundo, de acordo com a Bloomberg, atrás apenas do empresário Jeff Bezos.[3][4][5]
"""

if __name__ == "__main__":
    print("Hello LangChain!")

    summary_template = """"
        given the information {information} about a person from I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    print(chain.run(information=information))
