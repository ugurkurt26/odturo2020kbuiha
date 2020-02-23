# -*- coding: utf8 -*-




from __future__ import print_function
from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative
from pymavlink import mavutil # Needed for command message definitions
from ctypes import *
import math
import random
import numpy as np
from numpy.linalg import norm
import cv2,Queue, threading, time
import time
#Set up option parsing to get connection string
import argparse  

#connection_string = "127.0.0.1:14550"
#connection_string = '0.0.0.0:14550'
connection_string = 'udp:192.168.4.15:14550'
groundSpeed = 1

# Connect to the Vehicle
print('Connecting to vehicle on: %s' % connection_string)
vehicle = connect(connection_string, wait_ready=True)


def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]



colors = [tuple(255 * np.random.rand(3)) for _ in range(15)]
  

lib = CDLL("/home/nvidia/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.25, hier_thresh=.5, nms=.45):
    im = load_image(image.encode('utf-8'), 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res


class VideoCapture:

  def __init__(self, name):

    self.cap = cap
    self.q = Queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # Mümkün olan en yeni kareyi okur, sadece en son kareyi depolama.
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          
          # Önceki (işlenmemiş) kareleri hafızadan temizleme.
          self.q.get_nowait() 
        except Queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return 1,self.q.get()

def arm_and_takeoff(basetko,aTargetAltitude):
    
    """
    Arms vehicle and fly to aTargetAltitude.
    """
    
    print("Basic pre-arm checks")
    # Don't let the user try to arm until autopilot is ready
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)

        
    print("Arming motors")
    # Copter should arm in GUIDED mode
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:      
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    print(basetko)
    print(aTargetAltitude*0.95)
    vehicle.simple_takeoff(aTargetAltitude) # Take off to target altitude

    # Wait until the vehicle reaches a safe height before processing the goto (otherwise the command 
    #  after Vehicle.simple_takeoff will execute immediately).
    while True:
        print(" Altitude: {} , {}".format(vehicle.location.global_frame.alt - basetko,aTargetAltitude*0.95))      
        if vehicle.location.global_frame.alt-basetko>=aTargetAltitude*0.95: #Trigger just below target alt.
            print("Reached target altitude")
            break
        time.sleep(1)


"""
Convenience functions for sending immediate/guided mode commands to control the Copter.

The set of commands demonstrated here include:
* MAV_CMD_CONDITION_YAW - set direction of the front of the Copter (latitude, longitude)
* MAV_CMD_DO_SET_ROI - set direction where the camera gimbal is aimed (latitude, longitude, altitude)
* MAV_CMD_DO_CHANGE_SPEED - set target speed in metres/second.


The full set of available commands are listed here:
http://dev.ardupilot.com/wiki/copter-commands-in-guided-mode/
"""

def condition_yaw(heading, relative=False):
    """
    Send MAV_CMD_CONDITION_YAW message to point vehicle at a specified heading (in degrees).

    This method sets an absolute heading by default, but you can set the `relative` parameter
    to `True` to set yaw relative to the current yaw heading.

    By default the yaw of the vehicle will follow the direction of travel. After setting 
    the yaw using this function there is no way to return to the default yaw "follow direction 
    of travel" behaviour (https://github.com/diydrones/ardupilot/issues/2427)

    For more information see: 
    http://copter.ardupilot.com/wiki/common-mavlink-mission-command-messages-mav_cmd/#mav_cmd_condition_yaw
    """
    if relative:
        is_relative = 1 #yaw relative to direction of travel
    else:
        is_relative = 0 #yaw is an absolute angle
    # create the CONDITION_YAW command using command_long_encode()
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_CONDITION_YAW, #command
        0, #confirmation
        heading,    # param 1, yaw in degrees
        0,          # param 2, yaw speed deg/s
        1,          # param 3, direction -1 ccw, 1 cw
        is_relative, # param 4, relative offset 1, absolute angle 0
        0, 0, 0)    # param 5 ~ 7 not used
    # send command to vehicle
    vehicle.send_mavlink(msg)



def get_location_metres(original_location, dNorth, dEast):
    """
    Returns a LocationGlobal object containing the latitude/longitude `dNorth` and `dEast` metres from the 
    specified `original_location`. The returned LocationGlobal has the same `alt` value
    as `original_location`.

    The function is useful when you want to move the vehicle around specifying locations relative to 
    the current vehicle position.

    The algorithm is relatively accurate over small distances (10m within 1km) except close to the poles.

    For more information see:
    http://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters
    """
    earth_radius = 6378137.0 #Radius of "spherical" earth
    #Coordinate offsets in radians
    dLat = dNorth/earth_radius
    dLon = dEast/(earth_radius*math.cos(math.pi*original_location.lat/180))

    #New position in decimal degrees
    newlat = original_location.lat + (dLat * 180/math.pi)
    newlon = original_location.lon + (dLon * 180/math.pi)
    if type(original_location) is LocationGlobal:
        targetlocation=LocationGlobal(newlat, newlon,original_location.alt)
    elif type(original_location) is LocationGlobalRelative:
        targetlocation=LocationGlobalRelative(newlat, newlon,original_location.alt)
    else:
        raise Exception("Invalid Location object passed")
        
    return targetlocation;


def get_distance_metres(aLocation1, aLocation2):
    """
    Returns the ground distance in metres between two LocationGlobal objects.

    This method is an approximation, and will not be accurate over large distances and close to the 
    earth's poles. It comes from the ArduPilot test code: 
    https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
    """
    dlat = aLocation2.lat - aLocation1.lat
    dlong = aLocation2.lon - aLocation1.lon
    return math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5


def get_bearing(aLocation1, aLocation2):
    """
    Returns the bearing between the two LocationGlobal objects passed as parameters.

    This method is an approximation, and may not be accurate over large distances and close to the 
    earth's poles. It comes from the ArduPilot test code: 
    https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
    """	
    off_x = aLocation2.lon - aLocation1.lon
    off_y = aLocation2.lat - aLocation1.lat
    bearing = 90.00 + math.atan2(-off_y, off_x) * 57.2957795
    if bearing < 0:
        bearing += 360.00
    return bearing;



"""
Functions to move the vehicle to a specified position (as opposed to controlling movement by setting velocity components).

The methods include:
* goto_position_target_global_int - Sets position using SET_POSITION_TARGET_GLOBAL_INT command in 
    MAV_FRAME_GLOBAL_RELATIVE_ALT_INT frame
* goto_position_target_local_ned - Sets position using SET_POSITION_TARGET_LOCAL_NED command in 
    MAV_FRAME_BODY_NED frame
* goto - A convenience function that can use Vehicle.simple_goto (default) or 
    goto_position_target_global_int to travel to a specific position in metres 
    North and East from the current location. 
    This method reports distance to the destination.
"""

def goto_position_target_global_int(aLocation):
    """
    Send SET_POSITION_TARGET_GLOBAL_INT command to request the vehicle fly to a specified LocationGlobal.

    For more information see: https://pixhawk.ethz.ch/mavlink/#SET_POSITION_TARGET_GLOBAL_INT

    See the above link for information on the type_mask (0=enable, 1=ignore). 
    At time of writing, acceleration and yaw bits are ignored.
    """
    msg = vehicle.message_factory.set_position_target_global_int_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, # frame
        0b0000111111111000, # type_mask (only speeds enabled)
        aLocation.lat*1e7, # lat_int - X Position in WGS84 frame in 1e7 * meters
        aLocation.lon*1e7, # lon_int - Y Position in WGS84 frame in 1e7 * meters
        aLocation.alt, # alt - Altitude in meters in AMSL altitude, not WGS84 if absolute or relative, above terrain if GLOBAL_TERRAIN_ALT_INT
        0, # X velocity in NED frame in m/s
        0, # Y velocity in NED frame in m/s
        0, # Z velocity in NED frame in m/s
        0, 0, 0, # afx, afy, afz acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink) 
    # send command to vehicle
    vehicle.send_mavlink(msg)



def goto_position_target_local_ned(north, east, down):
    """	
    Send SET_POSITION_TARGET_LOCAL_NED command to request the vehicle fly to a specified 
    location in the North, East, Down frame.

    It is important to remember that in this frame, positive altitudes are entered as negative 
    "Down" values. So if down is "10", this will be 10 metres below the home altitude.

    Starting from AC3.3 the method respects the frame setting. Prior to that the frame was
    ignored. For more information see: 
    http://dev.ardupilot.com/wiki/copter-commands-in-guided-mode/#set_position_target_local_ned

    See the above link for information on the type_mask (0=enable, 1=ignore). 
    At time of writing, acceleration and yaw bits are ignored.

    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED, # frame
        0b0000111111111000, # type_mask (only positions enabled)
        north, east, down, # x, y, z positions (or North, East, Down in the MAV_FRAME_BODY_NED frame
        0, 0, 0, # x, y, z velocity in m/s  (not used)
        0, 0, 0, # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)    # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink) 
    # send command to vehicle
    vehicle.send_mavlink(msg)



def goto(dNorth, dEast, gotoFunction=vehicle.simple_goto):
    """
    Moves the vehicle to a position dNorth metres North and dEast metres East of the current position.

    The method takes a function pointer argument with a single `dronekit.lib.LocationGlobal` parameter for 
    the target position. This allows it to be called with different position-setting commands. 
    By default it uses the standard method: dronekit.lib.Vehicle.simple_goto().

    The method reports the distance to target every two seconds.
    """
    
    currentLocation = vehicle.location.global_relative_frame
    targetLocation = get_location_metres(currentLocation, dNorth, dEast)
    targetDistance = get_distance_metres(currentLocation, targetLocation)
    gotoFunction(targetLocation)
    
    #print "DEBUG: targetLocation: %s" % targetLocation
    #print "DEBUG: targetLocation: %s" % targetDistance

    remainingDistance=get_distance_metres(vehicle.location.global_relative_frame, targetLocation)
    while remainingDistance>=targetDistance*0.1: 
        #print "DEBUG: mode: %s" % vehicle.mode.name
        remainingDistance=get_distance_metres(vehicle.location.global_relative_frame, targetLocation)
	#print(str(remainingDistance))
        time.sleep(0.1)

def ortala():
    aaaaaaaaaaaaaaaaaaaaaaaaaa = logo_tespit(1)
    PIDSayac = 0
    print ("Ortalandı")    

def to_quaternion(roll = 0.0, pitch = 0.0, yaw = 0.0):
    """
    Convert degrees to quaternions
    """
    t0 = math.cos(math.radians(yaw * 0.5))
    t1 = math.sin(math.radians(yaw * 0.5))
    t2 = math.cos(math.radians(roll * 0.5))
    t3 = math.sin(math.radians(roll * 0.5))
    t4 = math.cos(math.radians(pitch * 0.5))
    t5 = math.sin(math.radians(pitch * 0.5))

    w = t0 * t2 * t4 + t1 * t3 * t5
    x = t0 * t3 * t4 - t1 * t2 * t5
    y = t0 * t2 * t5 + t1 * t3 * t4
    z = t1 * t2 * t4 - t0 * t3 * t5

    return [w, x, y, z]

def gogogo(waypoint,isInsin):
    remainingDistance=get_distance_metres(vehicle.location.global_frame, waypoint)
    while remainingDistance>=1:
        remainingDistance=get_distance_metres(vehicle.location.global_frame,waypoint)
        vehicle.simple_goto(waypoint, groundspeed=groundSpeed)
        time.sleep(0.1)
    if isInsin:
        vehicle.mode = VehicleMode("BRAKE")
        time.sleep(2)
        ortala()
        vehicle.mode = VehicleMode("LAND")
        while vehicle.armed: 
            time.sleep(0.5)


detected_objects = ['tr','stm','odtu','ort','H']
font = cv2.FONT_HERSHEY_SIMPLEX
net = load_net("/home/nvidia/darknet/odtu_tiny.cfg".encode('utf-8'), "/home/nvidia/darknet/odtu_final.weights".encode('utf-8'), 0)
meta = load_meta("/home/nvidia/darknet/odtu.data".encode('utf-8'))
cap = cv2.VideoCapture(0)
less = 100
cap = VideoCapture("_")
cumErrorX = 0
ortErrX = 0
cumErrorY = 0
ortErrY = 0
lastErrX = 0
lastErrY = 0
elapsedTime = 0

maxMesafe = math.sqrt((640/2**2)+(480/2**2))

PIDSayac = 0
PIDDogrulama = 10
def logo_tespit(oran):
    global PIDSayac
    global lastErrX
    global lastErrY
    global cumErrorX
    global ortErrX
    global cumErrorY
    global ortErrY
    ynlsSayac = 0
    frame_id = 0
    a = 0
    b = 0
    logo_sayac = [0,0,0,0,0]
    n_frame = 8
    ref_n_frame_axies = []
    ref_n_frame_label = []
    ref_n_frame_axies_flatten = []
    ref_n_frame_label_flatten = []
    label_cnt = 1
    frm_num = 1
    min_distance = 50
    before = time.time()
    while(True):  
        now = time.time()
        elapsedTime = now - before
        ret,img = cap.read()
        if ret == True:
            cur_frame_axies = []
            cur_frame_label = []
            cv2.imwrite('test.jpg',img)
            outputs = detect(net, meta, "test.jpg")

            if str(outputs) == "[]" :
		logo_sayac = [0,0,0,0,0]
                ynlsSayac = 0
                vehicle.mode = VehicleMode("BRAKE")
            
	    for color,output in zip(colors,outputs):
                text = output[0].decode('utf-8')
                x = int(output[2][0])
                y = int(output[2][1])
                fw = int(output[2][2])
                fh = int(output[2][3])
                w = int(fw/2)
                h = int(fh/2)
                acc = int(output[1] * 100)
                left = y - h
                top = x - w
                right = y + h
                bottom = x + w
                lbl = float('nan')
                if text in detected_objects:
                    if(len(ref_n_frame_label_flatten) > 0):
                        b = np.array([(x,y)])
                        a = np.array(ref_n_frame_axies_flatten)
                        distance = norm(a-b,axis=1)
                        min_value = distance.min()
                        if(min_value < min_distance):
                            idx = np.where(distance==min_value)[0][0]
                            lbl = ref_n_frame_label_flatten[idx]
                            # print(idx)
                    if(math.isnan(lbl)):
                        lbl = label_cnt
                        label_cnt += 1
                    cur_frame_label.append(lbl)
                    cur_frame_axies.append((x,y))
		    if text == "tr" and acc >= 60:
		        label = "TR"
                        logo_sayac[0]+=1
		    elif text == "stm" and acc >= 60:
			label = "STM"
                        logo_sayac[1]+=1
		    elif text == "odtu" and acc >= 60:
			label = "ODTU"
                        logo_sayac[2]+=1
		    elif text == "ort" and acc >= 60:
			label = "ORT"
                        logo_sayac[3]+=1
		    elif text == "H" and acc >= 60:
			label = "H"
                        logo_sayac[4]+=1
		    if acc < 40:
			if ynlsSayac > 5:
			    logo_sayac = [0,0,0,0,0]
                            ynlsSayac = 0
		   	    print(ynlsSayac)
		            vehicle.mode = VehicleMode("BRAKE")
			ynlsSayac+=1
		    errX = 640/2 - (top+bottom)/2
	            errY = 480/2 -(left+right)/2

                    #cv2.rectangle(img,(top,left),(bottom,right),color,2)
                    #cv2.putText(img,'{}{}-{}%'.format(text,lbl,acc),(top,left), font, 1,(255,255,255),2)
            if(len(ref_n_frame_axies) == n_frame):
                del ref_n_frame_axies[0]
                del ref_n_frame_label[0]
            ref_n_frame_label.append(cur_frame_label)
            ref_n_frame_axies.append(cur_frame_axies)
            ref_n_frame_axies_flatten = [a for ref_n_frame_axie in ref_n_frame_axies for a in ref_n_frame_axie]
            ref_n_frame_label_flatten = [b for ref_n_frame_lbl in ref_n_frame_label for b in ref_n_frame_lbl]
            #cv2.imshow('image',img)
            if b !=0:
	        if max(logo_sayac) >= 10:
                    if logo_sayac[0] >= 10:
			vehicle.mode = VehicleMode("GUIDED")
		        mesafe = math.sqrt((errX**2)+(errY**2))
			if mesafe/maxMesafe <= oran :
			    print(mesafe/maxMesafe)
			    PIDSayac+=1
			if PIDSayac >= PIDDogrulama :
			    cv2.destroyAllWindows()
			    return "TR"
		       	outX , ortErrX , cumErrorX = PID(errX,lastErrX,elapsedTime,cumErrorX,ortErrX)
			outY , ortErrY , cumErrorY = PID(errY,lastErrY,elapsedTime,cumErrorY,ortErrY)
			hareketFun(outX,outY)
		       	logo_sayac[1]=0
		       	logo_sayac[2]=0
		       	logo_sayac[3]=0
		       	logo_sayac[4]=0
		       	lastErrX = errX
		       	lastErrY = errY
                    elif logo_sayac[1] >= 10:
			vehicle.mode = VehicleMode("GUIDED")
		        mesafe = math.sqrt((errX**2)+(errY**2))
			if mesafe/maxMesafe <= oran :
			    print(mesafe/maxMesafe)
			    PIDSayac+=1
			if PIDSayac >= PIDDogrulama :
			    cv2.destroyAllWindows()
			    return "STM"
       		       	outX , ortErrX , cumErrorX = PID(errX,lastErrX,elapsedTime,cumErrorX,ortErrX)
			outY , ortErrY , cumErrorY = PID(errY,lastErrY,elapsedTime,cumErrorY,ortErrY)
			hareketFun(outX,outY)
			logo_sayac[0]=0
			logo_sayac[2]=0
			logo_sayac[3]=0
			logo_sayac[4]=0
		       	lastErrX = errX
		       	lastErrY = errY
                    elif logo_sayac[2] >= 10:
			vehicle.mode = VehicleMode("GUIDED")
		        mesafe = math.sqrt((errX**2)+(errY**2))
			if mesafe/maxMesafe <= oran :
			    print(mesafe/maxMesafe)
			    PIDSayac+=1
			if PIDSayac >= PIDDogrulama :
			    cv2.destroyAllWindows()
			    return "ODTU"
       		       	outX , ortErrX , cumErrorX = PID(errX,lastErrX,elapsedTime,cumErrorX,ortErrX)
			outY , ortErrY , cumErrorY = PID(errY,lastErrY,elapsedTime,cumErrorY,ortErrY)
			hareketFun(outX,outY)
			logo_sayac[0]=0
			logo_sayac[1]=0
			logo_sayac[3]=0
			logo_sayac[4]=0
		       	lastErrX = errX
		       	lastErrY = errY
                    elif logo_sayac[3] >= 10:
			vehicle.mode = VehicleMode("GUIDED")
			mesafe = math.sqrt((errX**2)+(errY**2))
			if mesafe/maxMesafe <= oran :
			    print(mesafe/maxMesafe)
			    PIDSayac+=1
			if PIDSayac >= PIDDogrulama :
			    cv2.destroyAllWindows()
			    return "ORT"
       		       	outX , ortErrX , cumErrorX = PID(errX,lastErrX,elapsedTime,cumErrorX,ortErrX)
			outY , ortErrY , cumErrorY = PID(errY,lastErrY,elapsedTime,cumErrorY,ortErrY)
			hareketFun(outX,outY)
			logo_sayac[0]=0
			logo_sayac[1]=0
			logo_sayac[2]=0
			logo_sayac[4]=0
		       	lastErrX = errX
		       	lastErrY = errY
                    elif logo_sayac[4] >= 10:
			vehicle.mode = VehicleMode("GUIDED")
			mesafe = math.sqrt((errX**2)+(errY**2))
			if mesafe/maxMesafe <= oran :
			    print(mesafe/maxMesafe)
			    PIDSayac+=1
			if PIDSayac >= PIDDogrulama :
			    cv2.destroyAllWindows()
			    return "H"
		       	outX , ortErrX , cumErrorX = PID(errX,lastErrX,elapsedTime,cumErrorX,ortErrX)
			outY , ortErrY , cumErrorY = PID(errY,lastErrY,elapsedTime,cumErrorY,ortErrY)
			hareketFun(outX,outY)
			logo_sayac[0]=0
			logo_sayac[1]=0
			logo_sayac[2]=0
			logo_sayac[3]=0
		       	lastErrX = errX
		       	lastErrY = errY
                    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        before = now
    cap.release()
    cv2.destroyAllWindows()


def PID (Err,lastErr,elapsedTime,cumError,ortErr):
    ortErr = (ortErr + Err)/2
    print("{}".format(Err))
    KP = 0.005 
    KD = 0.1 
    KI = 0 #0.000000000005
    cumError += Err * elapsedTime
    rateError = (Err - lastErr)/elapsedTime
    out = KP*Err + KI*cumError + KD*rateError
    return out , ortErr , cumError

def yoloMove(roll_angle = 0.0, pitch_angle = 0.0,
	yaw_angle = None, yaw_rate = 0.0, use_yaw_rate = True,thrust = 0.5):
    if yaw_angle is None:
        yaw_angle = vehicle.attitude.yaw
    msg = vehicle.message_factory.set_attitude_target_encode(
        0,1,1,0b00000000 if use_yaw_rate else 0b00000100,
        to_quaternion(roll_angle, pitch_angle, yaw_angle), 
        0, 0, math.radians(yaw_rate),thrust
    )
    vehicle.send_mavlink(msg)
    return {"res":'YoloMovement'}

maxAngle = 5
def hareketFun(outX,outY) :
    if outX>=maxAngle : outX = maxAngle
    if outX<=-maxAngle : outX = -maxAngle
    if outY>=maxAngle : outY = maxAngle
    if outY<=-maxAngle : outY = -maxAngle
    #print("{} ,  {}".format(outX,outY))
    yoloMove(pitch_angle = -outY , thrust = 0.5,roll_angle = -outX)

baseAlt = 0

def main():


	Yaw = vehicle.heading
	# Nokta 0
	delta =	Yaw+45
	N = 5.3033*math.cos(math.radians(delta))
	E = 5.3033*math.sin(math.radians(delta))
	nokta0 = [N,E]

	# Nokta 1
	delta =	Yaw+180
	N = 7.5*math.cos(math.radians(delta))
	E = 7.5*math.sin(math.radians(delta))
	nokta1 = [N,E]

	# Nokta 2
	delta =	Yaw+270
	N = 7.5*math.cos(math.radians(delta))
	E = 7.5*math.sin(math.radians(delta))
	nokta2 = [N,E]

	# Nokta 3
	delta =	Yaw
	N = 7.5*math.cos(math.radians(delta))
	E = 7.5*math.sin(math.radians(delta))
	nokta3 = [N,E]

	# Nokta 4 
	home = vehicle.location.global_frame 
        baseAlt = home.alt - 1 
	home.alt = home.alt + 2 
	noktalar = [nokta0,nokta1,nokta2,nokta3]
	#Arm and take of to altitude of 5 meters
        while not vehicle.mode.name == "GUIDED" :
	    time.sleep(0.5)
	    pass 

	arm_and_takeoff(baseAlt,4.5)

	print("TRIANGLE path using standard Vehicle.simple_goto()")

	print("Set groundspeed to %s m/s." %groundSpeed)
	vehicle.groundspeed=groundSpeed
	#baseAlt = -4

	adim  = 0
	hedef = 0
	algila_liste = ["STM","ODTU","ORT","H"]
	kayit_hedef = []
	kayit_gps= []
	#print("Yaw 100 absolute")
	condition_yaw(Yaw)
	isGeriDonus = False
	isGeriDonus1 = False
	donusNoktasi =0 
	while 1:
            condition_yaw(Yaw)
	    vehicle.groundspeed=groundSpeed
	    print("HEDEF : %s"%hedef)
	    if hedef == 4:
		print("Eve gidiyoreeee")
		gogogo(home,True)
		break;
	    if not isGeriDonus: 
		if isGeriDonus1:
		    isGeriDonus1 = False
		    if not kayit_gps[-1] == "":
		        gogogo(kayit_gps[-1],False)
	      
		nokta =noktalar[adim]
		goto(nokta[0], nokta[1], goto_position_target_global_int)
		vehicle.mode = VehicleMode("BRAKE")
		time.sleep(2)
		logo = logo_tespit(3)
		print(logo)
		PIDSayac = 0 
		if logo == "STM":
		    kayit_hedef.append(0)
		elif logo == "ODTU":
		    kayit_hedef.append(1)
		elif logo == "ORT":
		    kayit_hedef.append(2)
		elif logo == "H":
		    kayit_hedef.append(3) 
		kayit_gps.append(vehicle.location.global_frame)
		adim +=1
		if algila_liste[hedef] == logo:
		    vehicle.mode = VehicleMode("LAND")
		    while vehicle.armed: 
		        time.sleep(0.5)
		    hedef+=1
		    time.sleep(2)
		    arm_and_takeoff(baseAlt,4.5)
		    i =0
		    while i<len(kayit_hedef):
		        if kayit_hedef[i] == hedef:
		            isGeriDonus = True
		            donusNoktasi = i
		            break;
		    	i+=1
	    else:
		gogogo(kayit_gps[donusNoktasi],True)
		hedef+=1
		time.sleep(2)
		arm_and_takeoff(baseAlt,4.5)
		isGeriDonus = False
		isGeriDonus1 = True
		i =0
		while i<len(kayit_hedef):
		    if kayit_hedef[i] == hedef:
		        isGeriDonus = True
		        donusNoktasi = i
		        break;
		    i+=1
	    vehicle.mode = VehicleMode("GUIDED")
	    condition_yaw(Yaw)
	    vehicle.groundspeed=5



	time.sleep(2)
	print("Setting RTL mode...")
	vehicle.mode = VehicleMode("LAND")


	#Close vehicle object before exiting script
	while vehicle.armed: 
	    time.sleep(0.5)
	print("Close vehicle object")
	vehicle.close()

	# Shut down simulator if it was started.
	print("Completed")




if __name__ == "__main__":
    main()

