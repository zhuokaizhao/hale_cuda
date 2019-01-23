#
# Hale: support for minimalist scientific visualization
# Copyright (C) 2014, 2015  University of Chicago
#
# This software is provided 'as-is', without any express or implied
# warranty. In no event will the authors be held liable for any damages
# arising from the use of this software. Permission is granted to anyone to
# use this software for any purpose, including commercial applications, and
# to alter it and redistribute it freely, subject to the following
# restrictions:
#
# 1. The origin of this software must not be misrepresented; you must not
# claim that you wrote the original software. If you use this software in a
# product, an acknowledgment in the product documentation would be
# appreciated but is not required.
#
# 2. Altered source versions must be plainly marked as such, and must not be
# misrepresented as being the original software.
#
# 3. This notice may not be removed or altered from any source distribution.
#

LIB = libhale.a

#TEEM = $(HOME)/teem-install
# TEEM = /project/glk/rossc/code/teem-install
# zhuokai: changed Teem directory on dali
TEEM = /software/teem-trunk-el7-x86_64/include/
GLM = /software/glm-0.9-el6-x86_64/include/
GLFW = /software/glfw-3.1-el6-x86_64/include/

IPATH = -I$(TEEM) -I$(GLM) -I$(GLFW)

CXX = g++
AR = ar crs
CXXFLAGS = -Wall -std=c++11 -g

HDR = Hale.h
PRIV_HDR = privateHale.h
SRCS = enums.cpp globals.cpp utils.cpp Camera.cpp Viewer.cpp Program.cpp \
	Polydata.cpp Scene.cpp
OBJS = $(SRCS:.cpp=.o)

DEST = hale-install
DEST_LIB = $(DEST)/lib/$(LIB)
DEST_HDR = $(DEST)/include/$(HDR)

all: $(LIB)

$(OBJS): $(HDR) $(PRIV_HDR)

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $(IPATH) -o $@ $<

$(LIB): $(OBJS)
	$(AR) $(LIB) $(OBJS)

$(DEST_LIB): $(LIB)
	mkdir -p $(DEST)/lib
	cp $(LIB) $(DEST_LIB)

$(DEST_HDR): $(LIB) $(HDR) $(DEST_LIB)
	mkdir -p $(DEST)/include
	cp $(HDR) $(DEST_HDR)

install: $(DEST_LIB) $(DEST_HDR)

clean:
	rm -f $(OBJS) $(LIB)

clobber: clean
	rm -f $(DEST_LIB) $(DEST_HDR)
