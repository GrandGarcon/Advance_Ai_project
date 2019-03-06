import pygame
import sys
from math import *

#Declaration of constants
dx = 920
dy = 480
gLev = 370

red = (255,0,0)
sky = (180,225,255)
earth = (149,69,53)
star = (255,230,20)
grass = (0,127,50)
black = (0,0,0)
grey = (127,127,127)

#initialize physics
b_acc = 0.0
c_acc = 0.0
x = 0.0
xd = 0.0
h = 0.1
hd = 0.0
theta = 0.0
g = 0.000001
F = 0.000003
l = 150

#initialize display
pygame.init()
screen = pygame.display.set_mode((dx,dy))
clock = pygame.time.Clock()

left = False
right = False

while (True):
	dt = clock.tick(60)

	screen.fill(sky)
	pygame.draw.rect(screen, grass, (0,gLev+30,dx,dy-gLev+30), 0)
	pygame.draw.rect(screen, earth, (dx/4-30,gLev,dx/2+60,30), 0)
	pygame.draw.rect(screen, earth, (dx/4-60,gLev-30,30,60), 0)
	pygame.draw.rect(screen, earth, (3*dx/4+60 - 30,gLev-30,30,60), 0)

	xcor = int((x+1)*dx/2)
	ball = (int(xcor + l*cos(pi/2 - theta)), int((gLev-30) - l*sin(pi/2 - theta)))

	pygame.draw.rect(screen, grey, (xcor-30,gLev-30,60,30), 0)
	pygame.draw.circle(screen, red, ball, 20, 0)
	pygame.draw.line(screen, black, (xcor,gLev-15), ball, 3)

	for event in pygame.event.get():
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_ESCAPE:
				pygame.quit()
				sys.exit()
			if event.key == pygame.K_LEFT:
				left = True
			if event.key == pygame.K_RIGHT:
				right = True

		if event.type == pygame.KEYUP:
			if event.key == pygame.K_LEFT:
				left = False
			if event.key == pygame.K_RIGHT:
				right = False

	xdd = 0.0
	if left and not right and x > -0.5:
		xdd = -F
	elif not left and right and x < 0.5:
		xdd = F

	hd += (g*sin(theta)*cos(theta) - xdd*cos(theta)*cos(theta)) * dt

	xd += xdd * dt
	x += xd * dt
	h += hd * dt

	if x >= 0.5:
		x = 0.5
		xd = 0.0
	if x <= -0.5:
		x = -0.5
		xd = 0.0

	if h <= -1.0 or h >= 1.0:
		print("YOU LOST")
		pygame.quit()
		sys.exit()

	theta = asin(h)


	pygame.display.update()













