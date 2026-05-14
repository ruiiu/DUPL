def get_gpt4_ICE():
    example_1 = """
Hint: Please answer the question requiring an integer answer and provide the final value,
e.g., 1, 2, 3, at the end.\n
Question: Which number is missing?\n
Model response: The number missing in the sequence is 14.\n
Extracted answer: 14
"""

    example_2 = """
Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value,
e.g., 1.2, 1.3, 1.4, at the end.\n
Question: What is the fraction of females facing the camera?\n
Model response: The fraction of females facing the camera is 0.6,
which means that six out of ten females in the group are facing the camera.\n
Extracted answer: 0.6
"""

    example_3 = """
Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value,
e.g., 1.23, 1.34, 1.45, at the end.\n
Question: How much money does Luca need to buy a sour apple candy and a butter-scotch candy? (Unit: $)\n
Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.\n
Extracted answer: 1.45
"""

    example_4 = """
Hint: Please answer the question requiring a Python list as an answer and provide the final list,
e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.\n
Question: Between which two years does the line graph saw its maximum peak?\n
Model response: The line graph saw its maximum peak between 2007 and 2008.\n
Extracted answer: [2007, 2008]
"""

    example_5 = """
Hint: Please answer the question and provide the correct option letter only, e.g., A, B, C, D, at the end.\n
Question: What fraction of the shape is blue?\n
Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5\n
Model response: The correct answer is (B) 8/11.\n
Extracted answer: B
"""

    return [example_1, example_2, example_3, example_4, example_5]


def get_gpt4_score_ICE():
    example_1 = """
[Question]: Write the set of numbers represented on the number line in interval notation.
[Standard Answer]: (-2,1]
[Model_answer]: Extracted Answer: \\((-2, 1)\\)
Judgement: 0
""" # noqa

    example_2 = """
[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\nA:2\nB:2\u221a{{3}}\nC:\u221a{{3}}\nD:2\u221a{{2}}
[Standard Answer]: C
[Model_answer]: B:2\u221a{{3}}
Judgement: 0
""" # noqa

    example_3 = """
[Question]: Find the domain and range of the function f using interval notation.
[Standard Answer]: domain: [-4, 0) and range: (-3, 1]
[Model_answer]: Range: \\((-4, 1]\\)
Judgement: 0
""" # noqa

    example_4 = """
[Question]: As shown in the figure, circle O has a radius 1.0, if angle BAC = 60.0, then the length of BC is ()\nChoices:\n(A):2\n(B):2\u221a{{3}}\n(C):\u221a{{3}}\n(D):2\u221a{{2}}
[Standard Answer]: (C)
[Model_answer]: C
Judgement: 1
""" # noqa
    return [example_1, example_2, example_3, example_4]

def get_gpt4_chartqa_score_ICE():
    example_1 = """
[Question]: How many food item is shown in the bar graph?
[Standard Answer]: 15
[Model_answer]: Extracted Answer: 15 items.
Judgement: 1
""" # noqa

    example_2 = """
[Question]: What is the difference in value between Lamb and Corn?
[Standard Answer]: 8
[Model_answer]: 10
Judgement: 0
""" # noqa

    example_3 = """
[Question]: What was the 4th most popular emotion?
[Standard Answer]: Angry
[Model_answer]: Happy
Judgement: 0
""" # noqa

    example_4 = """
[Question]: What percent who think of President Donald Trump as Dangerous?
[Standard Answer]: 62
[Model_answer]: 62
Judgement: 1
""" # noqa
    return [example_1, example_2, example_3, example_4]

def get_gpt4_logicvista_score_ICE():
    example_1 = """
[Question]: Which of the boxes comes next? Choose from options A-E.
[Standard Answer]: E
[Model_answer]: Extracted Answer: E
Judgement: 1
""" # noqa

    example_2 = """
[Question]: The Golden Watch displays the time as 13:00. Select from A, B and C. (A) True (B)False (C)Insufficient Information
[Standard Answer]: A
[Model_answer]: B
Judgement: 0
""" # noqa

    example_3 = """
[Question]: What will girl on right write? Options are A. 13, B. 14, C. 15, D. 16
[Standard Answer]: B
[Model_answer]: D
Judgement: 0
""" # noqa

    example_4 = """
[Question]: Which of the choices can match the requirement Choose from A to D?
[Standard Answer]: B, C
[Model_answer]: B, C
Judgement: 1
""" # noqa
    return [example_1, example_2, example_3, example_4]

def get_gpt4_r1_onevision_score_ICE():
    example_1 = """
[Question]: What is the acceleration of a 10kg object under a net force of 20N? Use Newton's second law.
[Standard Answer]: 2 m/s^2
[Model_answer]: Extracted Answer: 2 m/s^2
Judgement: 1
"""  # noqa

    example_2 = """
[Question]: Given a DNA strand with the sequence AGCT, what is the complementary strand?
[Standard Answer]: TCGA
[Model_answer]: Extracted Answer: AGCT
Judgement: 0
"""  # noqa

    example_3 = """
[Question]: If the resistance is 10 ohms and the current is 2 amperes, what is the voltage across the resistor?
[Standard Answer]: 20 V
[Model_answer]: Extracted Answer: 20V
Judgement: 1
"""  # noqa

    example_4 = """
[Question]: What is the derivative of f(x) = x^2 + 3x + 5?
[Standard Answer]: 2x + 3
[Model_answer]: Extracted Answer: x^2 + 3x + 5
Judgement: 0
"""  # noqa

    return [example_1, example_2, example_3, example_4]


def get_gpt4_extract_ICE():
    example_1 = """
1.
Model response: 'Rounded to two decimal places, the perimeter of the sector is approximately:\n\n(-2, 1)'
Extracted Answer: (-2, 1)
""" # noqa

    example_2 = """
2.
Model response: 'at those points.\n\nTherefore, the correct option that represents the meaning of the intersection points of the graphs is:\n\nD. They give the solutions to the equation $f(t)=g(t)$.",'
Extracted Answer: D
""" # noqa

    example_3 = """
3.
Model response: ' at 1 (there's a closed circle at y = 1), the range in interval notation is \\((-4, 1]\\).\n\nFinal values:\nDomain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)'
Extracted Answer: Domain: \\((-3, 3]\\)\nRange: \\((-4, 1]\\)
""" # noqa

    example_4 = """
4.
Model response: 'As it stands, I cannot provide the correct option letter because there isn't enough information to solve for 'y'.'
Extracted Answer: null
""" # noqa

    example_5 = """
5.
Model response: 'Given that AB = 17.6 meters, we can now substitute into the equation:\n\nd = 17.6 / cos(38\u00b0)\n\nTherefore, to one decimal place, the distance d between Ned and Bart is approximately 22.3 meters.'
Extracted answer: 22.3
""" # noqa

    example_6 = """
6.
Model response:  have all the coefficients for the quadratic function:\n\\( f(x) = ax^2 + bx + c \\)\n\\( f(x) = -1x^2 - 2x + 1 \\)\n\nTherefore, the equation for the graphed function \\( f \\) is:\n\\( f(x) = -x^2 - 2x + 1 \\)"'
Extracted answer: f(x) = -x^2 - 2x + 1
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6]