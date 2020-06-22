#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 13:41:20 2020

@author: Amine
"""

import utilities as util


class Parent:

    def __init__(
            self,
            name='parent',
            city='Madrid',
            **kwargs):

        self.name = name
        self.city = city

        util.args_to_attributes(self, **kwargs)

    def show(self):

        print(self.name)    # Parent.D, child.D
        print(self.age)     # xxx       child.D --
        print(self.city)    # Parent.D, xxx     --
        print(self.wife)    # xxx       child.A --


class Child(Parent):

    def __init__(
            self,
            name='Zuina',
            age=18,
            **kwargs):

        self.name = name
        self.age = age

        super().__init__(name=name, **kwargs)

# %% Instantiate


A = Child(wife='Frida', city='Rabat')
A.show()
