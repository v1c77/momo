# -*- coding: utf-8 -*-

from momo import (
    Engine,
    DocumentBase,
    session_maker,
    Field,
)
from momo.validator import (
    String,
    Integer,
    Decimal,
)

engine = Engine(db='moen')
Session = session_maker(engine)


class Student(DocumentBase):
    __collection__ = 'moen'
    id = Field('_id', String, primary=True)
    name = Field(String(max_len=200), required=True)
    age = Field(Integer(min=0))
    wallet = Field(min=0, default=0)


if __name__ == '__main__':

    # type1
    # TODO(vici) draft weather use session
    std_1 = Student.create(name='yuhua', age='25')  # committed to db

    # type2
    std_2 = Student(name='yuhua')
    std_2.age = 123
    std_2.save()

    print(std_2.age)

    # transactions
    # Mongo version > 4.0 transaction default to True
    with Session(trans=True) as session:
        std_3 = Student.query(id='3')
        std_4 = Student.query(id='4')

        # trans 4D from std_3 to std_4
        std_3.wallet += 100
        # if std_2.wallet < 100, validate before commit will raise.
        std_4.wallet -= 100
