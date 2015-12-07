import collections, sys, os
from logic import *

############################################################
# Problem 6: (Extra Credit)
# Write a parser for our natural language interface.

from nlparser import GrammarRule, getCategoryProcessor

def createLanguageProcessor():
    # Defines a mapping from each in-domain word to its word class.
    # This automatically creates rules such as
    #   $Noun -> cat
    # with the string "cat" as the denotation
    categories = {
        'Noun': ['cat', 'tabby', 'dog', 'hound', 'dolphin', 'mammal', 'leg', 'foot', 'tail', 'fin'],
        'Name': ['Garfield', 'Pluto'],
        }
    for word in ['every', 'is', 'a', 'has', 'no', 'if', 'it', 'then', ',', '.', '?']:
        categories[word] = [word]
    return getCategoryProcessor(categories)

def createNLIGrammar():
    # Add your rules to the provided variable named rules.
    # Three examples are provided for you.
    # Please see nlparser.py for more information on the GrammarRule class.
    # IMPORTANT: Name all added variables '$x' (or '$y if necessary')
    rules = []

    # Parse if it's a question or statement.
    rules.append(GrammarRule('$ROOT', ['$Statement'], lambda args: ('tell', args[0])))
    rules.append(GrammarRule('$ROOT', ['$Question'], lambda args: ('ask', args[0])))
    rules.append(GrammarRule('$Statement', ['$Clause', '.'], lambda args: args[0]))
    rules.append(GrammarRule('$Question', ['$Clause', '?'], lambda args: args[0]))

    # (1) every X is a Y.
    rules.append(GrammarRule('$Clause', ['every', '$Noun', 'is', 'a', '$Noun'],
        lambda args: Forall('$x', Implies(Atom(args[0].title(), '$x'), Atom(args[1].title(), '$x')))))

    # (2) X is a Y.
    rules.append(GrammarRule('$Clause', ['$Name', 'is', 'a', '$Noun'],
        lambda args: Atom(args[1].title(), args[0].lower())))

    # (3) is X a Y?
    rules.append(GrammarRule('$Question', ['is', '$Clause-be', '?'],
        lambda args: args[0]))
    rules.append(GrammarRule('$Clause-be', ['$Name', 'a', '$Noun'],
        lambda args: Atom(args[1].title(), args[0].lower())))

    # BEGIN_YOUR_CODE (around 15 lines of code expected)
    
    # (1) Every X has a Y

    def functionOne(args):
        AtomX = Atom(args[0].title(), '$x')
        AtomY = Atom(args[1].title(), '$y')
        AtomHas = Atom('Has', '$x', '$y')
        return Forall('$x',
                 Implies(AtomX,
                         Exists('$y',
                         And(AtomY, AtomHas))))
    grammarRuleOne = GrammarRule('$Clause', ['every', '$Noun', 'has', 'a', '$Noun'], functionOne) 

    rules.append(grammarRuleOne)

    # (2) No X has a Y

    def functionTwo(args):
        AtomX = Atom(args[0].title(), '$x')
        AtomY = Atom(args[1].title(), '$y')
        AtomHas = Atom('Has', '$x','$y')
        return Forall('$x',
              Implies(AtomX, Forall('$y',
                                   And(Not(AtomHas), AtomY)
                                   ))
              )

    grammarRuleTwo = GrammarRule('$Clause', ['no', '$Noun', 'has', 'a', '$Noun'], functionTwo)
    rules.append(grammarRuleTwo)

    # (3) If a X has a Y, then it has a Z
    
    def functionThree():
        AtomX = Atom(args[0].title(), '$x')
        AtomY = Atom(args[1].title(), '$y')
        AtomHas = Atom('has', '$x','$y')
        return Forall('$x',
                        Implies(AtomX,
                                Forall('$y',
                                      And(Not(AtomHas), AtomY)
                                      )
                               )
                     )

    grammarRuleThree = GrammarRule('$Clause', ['if', 'a','$Noun','has','a','$Noun',',','then','it','has','a','$Noun'], functionThree)

    rules.append(grammarRuleThree)

    # END_YOUR_CODE
    return rules
