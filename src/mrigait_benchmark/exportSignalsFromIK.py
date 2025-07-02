"""
    Credits to Micheal Raitor, Nick Bianco
    From the simTK forums: https://simtk.org/plugins/phpBB/viewtopicPhpbb.php?f=91&t=16454&p=0&start=0&view=
"""

import opensim as osim
import numpy as np
import pandas as pd


def get_signal(osim_path, ik_path):
    """
    This function computes the center of mass position, velocity, kinetic energy,
    angular momentum, and central inertia of a model given the model path and the
    inverse kinematics (IK) path. It saves the results in a CSV file.
    :param osim_path: Path to the OpenSim model file.
    :param ik_path: Path to the IK file.
    :return: outputs
    """
    # Import the model.
    modelBase = osim.Model(osim_path)
    modelBase.initSystem()

    # We are going to replace the joints with locked coordinates with WeldJoints below.
    # First, we need to remove the actuators in the model associated with the locked coordinates,
    # otherwise the ModelProcessor operation below will throw an error.
    forceSet = modelBase.updForceSet()
    forceSet.remove(forceSet.getIndex('wrist_flex_r'))
    forceSet.remove(forceSet.getIndex('wrist_flex_l'))
    forceSet.remove(forceSet.getIndex('wrist_dev_r'))
    forceSet.remove(forceSet.getIndex('wrist_dev_l'))

    # Use a ModelProcessor to replace the locked coordinates with WeldJoints. This will
    # prevent issues (eg. NaNs) when computing the quantities below.
    modelProcessor = osim.ModelProcessor(modelBase)
    jointsToWeld = ['subtalar_r', 'subtalar_l', 'mtp_r', 'mtp_l', 'radius_hand_r', 'radius_hand_l']
    modelProcessor.append(osim.ModOpReplaceJointsWithWelds(jointsToWeld))
    model = modelProcessor.process()
    model.initSystem()
    table = osim.TimeSeriesTable(ik_path)

    # We need to add two columns for coordinates associated with the patellofemoral joint,
    # which are enforced by the CoordinateCouplerConstraints in the model. If we don't add
    # these columns, NaNs will appear in the state for these coordinates.
    table.appendColumn('knee_angle_r_beta', table.getDependentColumn('knee_angle_r'))
    table.appendColumn('knee_angle_l_beta', table.getDependentColumn('knee_angle_l'))

    # Create a TableProcessor so that we can update the column labels
    # to full paths based on the model.
    tableProcessor = osim.TableProcessor(table)
    tableProcessor.append(osim.TabOpUseAbsoluteStateNames())
    coordinates = tableProcessor.process(model)

    # Now, let's spline the coordinates so we can compute coordinate speeds.
    time = np.array(coordinates.getIndependentColumn())
    coordinateSplines = osim.GCVSplineSet(coordinates)
    for ispline in range(coordinateSplines.getSize()):

        # Get the coordinate label.
        label = coordinates.getColumnLabel(ispline)

        # Skip if we're at a locked coordinate.
        if ('subtalar' in label) or ('mtp' in label) or ('wrist' in label):
            continue

        # Get the spline for the coordinate at the index 'ispline'.
        spline = coordinateSplines.getGCVSpline(ispline)

        # Create an empty vector to fill in the coordinate speed values.
        speed = osim.Vector(coordinates.getNumRows(), 0.0)

        # We need this vector to tell the calcDerivative() function that we want
        # the first derivative (vector of length 1) with respect to time (index 0;
        # is our only input argument anyway).
        derivComponents = osim.StdVectorInt()
        derivComponents.push_back(0)
        for i in range(coordinates.getNumRows()):
            # Construct the actual input vector (again, just time).
            x = osim.Vector(1, time[i])
            # Calculate the coordinate speed from the derivative of the coordinate
            # value.
            speed[i] = spline.calcDerivative(derivComponents, x)

        # Add the speed column to the table. Create a new label using the coordinate path
        # and replace 'value' with 'speed'.
        coordinates.appendColumn(label.replace('value', 'speed'), speed)
    # Now that we have a table with the coordinate values and speeds, let's convert
    # it to a StatesTrajectory so we can compute stuff from the model. Since we don't
    # have all of the states from the model in the table, we need to set 'allowMissingColumns'
    # to True.
    allowMissingColumns = True
    # This flag allows us to ignore the columns associated with the locked coordinates, which
    # we don't apply to the model anymore
    allowExtraColumns = True
    # This flag tells the model to 'assemble', which means it will try to enforce any kinematic
    # constraints in the model. Since we set the knee_angle_r_beta and knee_angle_l_beta coordinates
    # above to be equal to the knee_angle_r and knee_angle_l coordinates, respectively, we can safely
    # set this flag to False. (Setting it to True should produce the same results, but is much slower).
    assemble = False
    statesTraj = osim.StatesTrajectory.createFromStatesTable(model, coordinates,
                                                             allowMissingColumns, allowExtraColumns, assemble)

    # Now we can finally compute stuff! Note that since we don't have forces, we can't
    # actually compute center of mass acceleration this way, but we can compute everything else.
    # (There is a way to compute accelerations, I can follow up with you later about it).
    numTimes = statesTraj.getSize()
    columns = ['com_pos_x', 'com_pos_y', 'com_pos_z', 'com_vel_x', 'com_vel_y', 'com_vel_z', 'kinetic_energy',
               'angular_momentum_x', 'angular_momentum_y', 'angular_momentum_z', 'central_inertia_00',
               'central_inertia_11', 'central_inertia_22']
    outputs = pd.DataFrame(columns=columns, index=range(numTimes))

    for itime in range(numTimes):
        # Get the current SimTK::State and realize to Stage::Velocity, which is the
        # minimum stage required to compute the quantities below.
        state = statesTraj.get(itime)
        model.realizeVelocity(state)

        # Calculate the center of mass position and velocity.
        #com_pos = model.calcMassCenterPosition(state)
        #com_vel = model.calcMassCenterVelocity(state)
        #outputs.loc[itime, 'com_pos_x'] = com_pos[0]
        #outputs.loc[itime, 'com_pos_y'] = com_pos[1]
        #outputs.loc[itime, 'com_pos_z'] = com_pos[2]
        #outputs.loc[itime, 'com_vel_x'] = com_vel[0]
        #outputs.loc[itime, 'com_vel_y'] = com_vel[1]
        #outputs.loc[itime, 'com_vel_z'] = com_vel[2]

        # Calculate the kinetic energy.
        outputs.loc[itime, 'kinetic_energy'] = model.calcKineticEnergy(state)

        ## For whole-body angular momentum and inertia, we need to go down into Simbody.
        #matter = model.getMatterSubsystem()
        #momentum = matter.calcSystemCentralMomentum(state)
        ## 'momentum' is a SpatialVec containing the angular and linear momentum in a pair of Vec3's.
        ## The first Vec3 is the angular component.
        #angular_momentum = momentum.get(0)
        #outputs.loc[itime, 'angular_momentum_x'] = angular_momentum[0]
        #outputs.loc[itime, 'angular_momentum_y'] = angular_momentum[1]
        #outputs.loc[itime, 'angular_momentum_z'] = angular_momentum[2]
#
        ## The central inertia is the inertia tensor of the system about the center of mass.
        ## We'll just store the moments of inertia (i.e., the diagonal elements).
        #inertia = matter.calcSystemCentralInertiaInGround(state)
        #moments = inertia.getMoments()
        #outputs.loc[itime, 'central_inertia_00'] = moments[0]
        #outputs.loc[itime, 'central_inertia_11'] = moments[1]
        #outputs.loc[itime, 'central_inertia_22'] = moments[2]


    # Save the outputs to a CSV file.
    return outputs
