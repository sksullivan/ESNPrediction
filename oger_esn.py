# -*- coding: utf-8 -*-
"""
A minimalistic ESN demo with Makey-Glass delay 17 data using Oger toolbox
from http://reservoir-computing.org/oger .
by Mantas LukoÅ¡eviÄius 2012
http://minds.jacobs-university.de/mantas
"""
from numpy import *
from matplotlib.pyplot import *
import Oger
import mdp
import csv
from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
    FileTransferSpeed, FormatLabel, Percentage, \
    ProgressBar, ReverseBar, RotatingMarker, \
    SimpleProgress, Timer
from multiprocessing import Pool,Value

dimensions = Value('d',1.0)
processed = Value('i',0)
tasks = []
r = None
meta_results = []
pbar = None

def do_task(params):
	n = params[0]
	i = params[1]
	b = params[2]
	s = params[3]
	l = params[4]
	res = r.init_res(N=n,leak_rate=l,in_scale=i,bias_scale=b,spec_rad=s)
	flow = r.make_flow(res)
	# import pdb; pdb.set_trace()
	# print 1
	Y = r.run_trial(res,flow,r.data)
	# print 2
	mse = r.compute_error(r.data,Y)
	# print 3
	meta_results.append([n,i,b,s,l,mse])
	processed.value += 1
	print "%.2f" % (processed.value/dimensions.value * 100,)

class res_driver():
	def init_data(self):
		data = genfromtxt('forex.csv',delimiter=';',converters={0: lambda (x):1.0})
		data = data[:,1]
		Oger.utils.make_inspectable(Oger.nodes.LeakyReservoirNode)
		return data

	def init_res(self,N=1000,leak_rate=0.3,in_scale=0.5,bias_scale=0.5,spec_rad=0.2):
		# generate the ESN reservoir
		inSize = outSize = 1
		resSize = N

		random.seed(42)
		reservoir = Oger.nodes.LeakyReservoirNode(output_dim=resSize, leak_rate=leak_rate, \
	    	input_scaling=in_scale, bias_scaling=bias_scale, spectral_radius=spec_rad, reset_states=False) 

		# Tell the reservoir to save its states for later plotting 
		return reservoir

	def make_flow(self,reservoir):
		# create the output   
		reg = 1e-8   
		readout = Oger.nodes.RidgeRegressionNode( reg )
		flow = Oger.nodes.FreerunFlow([reservoir, readout], freerun_steps=self.testLen)
		return flow

	def run_trial(self,reservoir,flow,data):
		# train
		flow.train([[], [[data[0:self.trainLen+1,None]]]])

		# save states for plotting
		X = reservoir.inspect()[0]

		# run in a generative mode
		Y = flow.execute(array(data[self.trainLen-self.initLen:self.trainLen+self.testLen+1,None]))
		# discard the first elements (just a numbering convention)
		Y = Y[self.initLen+1:]
		return Y

	def compute_error(self,data,Y):
		# compute MSE for the first errorLen time steps
		self.errorLen = 500
		mse = sum( square( data[self.trainLen+1:self.trainLen+self.errorLen+1] - Y[0:self.errorLen,0] ) ) / self.errorLen
		return mse

	def vis_pred(self,data,Y):
		figure(1).clear()
		plot( data[self.trainLen+1:self.trainLen+self.testLen+1], 'g' )
		plot( Y )
		title('Target and generated signals $y(n)$')
		show()
	# plot some signals
	#figure(1).clear()
	#plot( data[trainLen+1:trainLen+testLen+1], 'g' )
	#plot( Y )
	#title('Target and generated signals $y(n)$')

	#figure(2).clear()
	#plot( X[initLen:initLen+200,0:20] )
	#title('Some reservoir activations $\mathbf{x}(n)$')

	def __init__(self):
		global tasks
		global dimensions
		global processed
		global pbar
		global meta_results

		self.trainLen = 2000
		self.testLen = 2000
		self.initLen = 100

		self.data = self.init_data()

		

		# plot some of it
		# figure(10).clear()
		# plot(data[0:1000])
		# self.vis_pred(data,Y)


		#figure(3).clear()
		#bar( range(1+resSize), readout.beta[:,0] )
		#title('Output weights $\mathbf{W}^{out}$')

	def start(self):
		global tasks
		global dimensions
		global processed
		global pbar
		global meta_results

		param_ranges_small = {
			'N': arange(100,300,100),
			'in_scale': arange(0.1,0.3,0.1),
			'bias_scale': arange(0.0,0.2,0.1),
			'spec_rad': arange(0.2,0.4,0.1),
			'leak_rate': arange(0.0,0.2,0.1),	
		}

		param_ranges = {
			'N': arange(100,2100,500),
			'in_scale': arange(0.1,1,0.1),
			'bias_scale': arange(0.0,0.5,0.1),
			'spec_rad': arange(0.2,2,0.1),
			'leak_rate': arange(0.0,0.5,0.1),	
		}

		# Generate all param combinations
		for param,p_range in param_ranges.iteritems():
			dimensions.value *= p_range.shape[0]
		print dimensions.value

		pool = Pool(6)
		meta_results = []
		for n in param_ranges['N']:
			for i in param_ranges['in_scale']:
				for b in param_ranges['bias_scale']:
					for s in param_ranges['spec_rad']:
						for l in param_ranges['leak_rate']:
							tasks.append([n,i,b,s,l])

		pool.map(do_task,tasks)
		print('writing')
		fl = open('results_big.csv', 'w')
		writer = csv.writer(fl)
		writer.writerow(['N', 'in_scale', 'bias_scale', 'spec_rad', 'leak_rate', 'mse'])
		for values in meta_results:
			writer.writerow(values)
		fl.close()

r = res_driver()
r.start()
