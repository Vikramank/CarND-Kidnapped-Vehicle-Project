/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include <cmath>
using namespace std;
//random engine
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[])
{

	// number of particles
	num_particles = 1;
	// noise for initial parameters
	normal_distribution<double> dist_x(0, std[0]);
	normal_distribution<double> dist_y(0, std[1]);
	normal_distribution<double> dist_theta(0, std[2]);
	//intialising weights and particles
	weights = vector<double>(num_particles);
	particles = vector<Particle>(num_particles);

	for (int i = 0; i < num_particles; i++)
	{
		Particle p;
		p.id = i;
		p.x = x + dist_x(gen);
		p.y = y + dist_y(gen);
		p.theta = theta + dist_theta(gen);
		p.weight = 1.0;
		//particles[i] = p;
		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	double vy = velocity / yaw_rate;
	double ydt = delta_t * yaw_rate;
	double vdt = delta_t * velocity;
	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);
	for (int i = 0; i < num_particles; i++)
	{

		Particle p = particles[i];
		// predict x,y for t+1
		if (fabs(yaw_rate) > 0.001)
		{
			p.x += vy * (sin(p.theta + ydt) - sin(p.theta));
			p.y += vy * (cos(p.theta) - cos(p.theta + ydt));
			p.theta += yaw_rate * delta_t;
		}
		else
		{
			p.x += vdt * cos(p.theta);
			p.y += vdt * sin(p.theta);
		}
		//p.theta += ydt;

		// adding noise parameters
		p.x += noise_x(gen);
		p.y += noise_y(gen);
		p.theta += noise_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations)
{
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i = 0; i < observations.size(); i++)
	{
		int map_id = -1;
		double min_dist = std::numeric_limits<double>::max();
		// grab current observation
		LandmarkObs obs = observations[i];

		for (int j = 0; j < predicted.size(); j++)
		{
			LandmarkObs pred = predicted[j];
			double distance = sqrt((pred.x - obs.x) * (pred.x - obs.x) + (pred.y - obs.y) * (pred.y - obs.y));
			if (distance < min_dist)
			{
				min_dist = distance;
				map_id = pred.id;
			}
		}

		observations[i].id = map_id;
		//cout<<map_id<<endl;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
								   std::vector<LandmarkObs> observations, Map map_landmarks)
{
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i = 0; i < num_particles; i++)
	{
		Particle p = particles[i];
		//step1 : Translating local coordinates to map coordinates
		std::vector<LandmarkObs> observations_map;
		LandmarkObs observation_transformed;
		for (int j = 0; j < observations.size(); j++)
		{

			LandmarkObs observation = observations[j];
			//translation plus rotation
			observation_transformed.x = observation.x * cos(p.theta) - observation.y * sin(p.theta) + p.x;
			observation_transformed.y = observation.x * sin(p.theta) + observation.y * cos(p.theta) + p.y;
			//appending this observation
			observations_map.push_back(observation_transformed);
		}

		//collecting all the landmarks within the range of the sensor
		std::vector<LandmarkObs> predicted_landmarks;
		for (int k = 0; k < map_landmarks.landmark_list.size(); k++)
		{
			Map::single_landmark_s landmark = map_landmarks.landmark_list[k];
			double distance = sqrt((p.x - landmark.x_f) * (p.x - landmark.x_f) + (p.y - landmark.y_f) * (p.y - landmark.y_f));

			if (distance < sensor_range)
			{
				LandmarkObs prediction;
				prediction.id = landmark.id_i;
				prediction.x = landmark.x_f;
				prediction.y = landmark.y_f;
				predicted_landmarks.push_back(prediction);
			}
		}

		// step 2: Choosing closest landmark
		dataAssociation(predicted_landmarks, observations_map);

		//step3 :Calculating probability for each observation and multpliying them to find total probability
		p.weight = 1.0;
		double denominator = 2 * M_PI * std_landmark[0] * std_landmark[1];
		double x_deno = 2 * pow(std_landmark[0], 2);
		double y_deno = 2 * pow(std_landmark[1], 2);
		for (int l = 0; l < observations_map.size(); l++)
		{
			LandmarkObs obs = observations_map[l];
			LandmarkObs mu = predicted_landmarks[obs.id];
			double mu_x;
			double mu_y;

			for (unsigned int m = 0; m < predicted_landmarks.size(); m++)
			{
				if (predicted_landmarks[m].id == obs.id)
				{
					mu_x = predicted_landmarks[m].x;
					mu_y = predicted_landmarks[m].y;
				}
			}

			double x_diff = obs.x - mu_x;
			double y_diff = obs.y - mu_y;

			
			cout << p.weight;
			p.weight *= exp(-(pow(x_diff, 2) / x_deno) - (pow(y_diff, 2) / y_deno)) / denominator;

			
		}

		weights[i] = p.weight;
	}
}

void ParticleFilter::resample()
{

	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	//implementing resampling wheel
	//new particles
	vector<Particle> new_particles(particles.size());
	//current weights
	vector<double> weights;
	for (int i = 0; i < num_particles; i++)
	{
		weights.push_back(particles[i].weight);
	}
	//choosing random index

	uniform_int_distribution<int> uniintdist(0, num_particles - 1);
	int index = uniintdist(gen);

	// maximum weight
	double max_weight = *max_element(weights.begin(), weights.end());

	// uniform random distribution
	uniform_real_distribution<double> unirealdist(0.0, 1.0);

	double beta = 0.0;

	// resampling wheel
	for (int i = 0; i < num_particles; i++)
	{
		beta += unirealdist(gen) * 2.0 * max_weight;

		while (weights[index] < beta)
		{
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		new_particles.push_back(particles[index]);
	}

	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
