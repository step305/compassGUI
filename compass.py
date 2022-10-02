import tkinter
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import ticker

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import multiprocessing as mp
import time
import cv2
from PIL import Image, ImageTk
from subprocess import check_output
import config
from scipy.optimize import curve_fit
import json
import base64
import os

# SLAM_COMMAND = '/home/step305/SLAM_NANO/slam_start.sh &'
SLAM_COMMAND = '/home/sergey/SLAM_NANO/slam_start.sh &'
# FIFO_PATH = '/home/step305/SLAM_FIFO.tmp'
FIFO_PATH = '/home/sergey/SLAM_FIFO.tmp'

periodic_update_id = ''
screen_width = 0
screen_height = 0

maytagging_first_run = True
maytagging_proc = None
maytagging_yaw_prev = 0
maytagging_adc_prev = 0

caruseling_run_cnt = 0
caruseling_proc = None
earth_meas_hist = []

MAYTAGGING_WAIT_PERIOD = 20
EartNorthComponent = 11.7

plot_x = []
plot_y = []

t_start_accumuate = 0

data_queue = mp.Queue(10)
stop_event = mp.Event()
next_event = mp.Event()
stop_event.clear()
next_event.clear()

azimuth = np.NAN

adc_sum = []
yaw_sum = []


def get_pid(name):
    try:
        ret = int(check_output(["pidof", name]))
    except Exception as e:
        ret = -100
    return ret


def config_plot_style(ax):
    ax.set_facecolor(config.PLOT_BG_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    y_fmt = ticker.FormatStrFormatter('%0.2f')
    ax.set_xticklabels(ax.get_xticks(), fontsize=10)
    ax.set_yticklabels(ax.get_yticks(), fontsize=10)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_major_formatter(y_fmt)
    ax.yaxis.set_major_formatter(y_fmt)
    ax.xaxis.set_minor_formatter(y_fmt)
    ax.yaxis.set_minor_formatter(y_fmt)


def fit_f(x, p1, p2, p3):
    return p1 + p2 * np.cos((p3 + x) * np.pi / 180)


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def draw_data(canvas, f_plot, panel, data):
    global periodic_update_id
    global azimuth
    global yaw_sum
    global adc_sum
    global plot_x
    global plot_y

    if stop_event.is_set():
        return
    if not data.empty():
        pack = data.get()
        slam_data, frame = pack

        if not slam_data is None:
            azi = (slam_data['yaw'] + azimuth) % 360.0
            if azi > 180:
                azi = azi - 360
            elif azi < -180:
                azi = azi + 360
            azi_val = '...' if np.isnan(azimuth) else '{:0.2f}'.format(azi)
            lbl_val = '\nEuler angles [deg]:\n{:0.1f},\t{:0.1f},\t{:0.1f}' \
                      '\nAzimuth [deg]:\n{:}' \
                      '\nBias [dph]:\n{:0.1f},\t{:0.1f},\t{:0.1f}'.format(slam_data['yaw'],
                                                                          slam_data['pitch'],
                                                                          slam_data['roll'],
                                                                          azi_val,
                                                                          slam_data['bw'][0],
                                                                          slam_data['bw'][1],
                                                                          slam_data['bw'][2])
            lbl_text.set(lbl_val)
            if next_event.is_set():
                if time.time() < t_start_accumuate + config.POINT_CARUSELING_DURATION:
                    yaw_sum.append(slam_data['yaw'])
                    adc_sum.append(slam_data['adc'])
                else:
                    yaw_point = np.unwrap(np.array(yaw_sum) / 180.0 * np.pi).mean() / np.pi * 180.0
                    adc_point = np.array(adc_sum).mean()
                    plot_x.append(yaw_point)
                    plot_y.append(adc_point)
                    print(plot_x)
                    print(plot_y)
                    adc_sum = []
                    yaw_sum = []
                    next_event.clear()
                    next_button['state'] = tkinter.NORMAL
                    f_plot.clear()
                    if polar_view.get() == 1:
                        ro, phi = cart2pol(np.array(plot_x), np.array(plot_y))
                        f_plot.polar(phi, ro, 'ro')
                    else:
                        f_plot.plot(np.array(plot_x), np.array(plot_y), 'ro')
                    if len(plot_y) >= 3:
                        try:
                            x = np.array(plot_x)
                            y = np.array(plot_y)
                            popt, pcov = curve_fit(fit_f, x, y, p0=(0.0, 10.2, 0))
                            azimuth = popt[2]
                            if popt[1] < 0:
                                azimuth += 180
                            xf = np.linspace(np.min(x), np.max(x), 100)
                            yf = fit_f(xf, popt[0], popt[1], popt[2])
                            if polar_view.get() == 1:
                                ro, phi = cart2pol(xf, yf)
                                f_plot.polar(phi, ro, 'b*')
                            else:
                                f_plot.plot(xf, yf, 'b-')
                            f_plot.legend(['phase = ', '{:0.2f}deg'.format(azimuth)])
                        except Exception as e:
                            print('Invalid arccos', e)
        config_plot_style(f_plot)
        canvas.draw()

        if not frame is None:
            w = config.VIDEO_PANEL_SIZE[0] * screen_width
            h = config.VIDEO_PANEL_SIZE[1] * screen_height
            frame = cv2.imdecode(frame, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(frame, (int(w), int(h)))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=im)
            if panel is None:
                panel = tkinter.Label(image=imgtk)
                panel.image = imgtk
                panel.place(x=screen_width - 3 * config.BORDERS_WIDTH - w, y=config.BORDERS_WIDTH)
                panel['borderwidth'] = 5
                panel['relief'] = 'raised'
            else:
                panel.configure(image=imgtk)
                panel.image = imgtk
                panel['borderwidth'] = 5
                panel['relief'] = 'flat'

        # w = screen_width * config.PLOT_PANEL_SIZE[0]
        # h = screen_height * config.PLOT_PANEL_SIZE[1]
        # polar_chack_box.place(x=config.BORDERS_WIDTH, y=2 * config.BORDERS_WIDTH + h + 10,
        #                      height=40, width=int(w / 3))
    if not stop_event.is_set():
        periodic_update_id = root.after(ms=5, func=lambda: draw_data(canvas, f_plot, panel, data))


def data_source(stop, queue):
    FIFO = FIFO_PATH
    cnt = 0
    max_cnt = 10
    heading_sum = []
    roll_sum = []
    pitch_sum = []
    bw_sum = [0, 0, 0]
    sw_sum = [0, 0, 0]
    crh_sum = 0
    # os.system(SLAM_COMMAND)
    print(SLAM_COMMAND)

    while not stop.is_set():
        try:
            with open(FIFO) as fifo:
                for line in fifo:
                    if stop.is_set():
                        break
                    try:
                        packet = json.loads(line)
                        if cnt == max_cnt:
                            package = {'yaw': np.unwrap(np.array(heading_sum) /180.0 * np.pi).mean() / np.pi * 180.0,
                                       'pitch': np.unwrap(np.array(pitch_sum) /180.0 * np.pi).mean() / np.pi * 180.0,
                                       'roll': np.unwrap(np.array(roll_sum) /180.0 * np.pi).mean() / np.pi * 180.0,
                                       'bw': [i / cnt for i in bw_sum],
                                       'sw': [i / cnt for i in sw_sum],
                                       'adc': crh_sum / cnt
                                       }
                            if not queue.full():
                                queue.put((package, None))
                            heading_sum = []
                            roll_sum = []
                            pitch_sum = []
                            bw_sum = [0, 0, 0]
                            crh_sum = 0
                            cnt = 0
                        else:
                            cnt = cnt + 1
                            heading_sum.append(packet['yaw'])
                            roll_sum.append(packet['roll'])
                            pitch_sum.append(packet['pitch'])
                            bw_sum = [x + y for x, y in zip(bw_sum, packet['bw'])]
                            sw_sum = [x + y for x, y in zip(sw_sum, packet['sw'])]
                            crh_sum += packet['adc']
                        if packet['frame'] == "None":
                            pass
                        else:
                            pass
                            buf_decode = base64.b64decode(packet['frame'])
                            jpg = np.fromstring(buf_decode, np.uint8)
                            if not queue.full():
                                queue.put((None, jpg))
                    except Exception as e:
                        print('Some error')
                        print(e)
        except Exception as e:
            print()
            print(e)
            print('Done!')
            break


def start_record_point():
    global t_start_accumuate
    next_button['state'] = tkinter.DISABLED
    t_start_accumuate = time.time()
    next_event.set()


def close_callback():
    stop_event.set()
    root.after_cancel(periodic_update_id)
    data_queue.close()
    time.sleep(0.3)
    data_source_proc.terminate()
    data_source_proc.join()
    root.destroy()


if __name__ == '__main__':
    root = tkinter.Tk()
    root.geometry('30x30')
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry('{}x{}'.format(screen_width, screen_height))
    root.configure(bg=config.SCREEN_BG_COLOR)
    root.attributes('-fullscreen', True)

    lbl_text = tkinter.StringVar()
    lbl = tkinter.Label(root, textvariable=lbl_text, relief=tkinter.GROOVE,
                        bg=config.SCREEN_BG_COLOR, fg='white',
                        font='Arial, 28', bd=0,
                        justify=tkinter.LEFT)
    lbl_text.set('SLAM data!')
    w = config.VIDEO_PANEL_SIZE[0] * screen_width
    h = config.VIDEO_PANEL_SIZE[1] * screen_height
    lbl.place(x=screen_width - config.BORDERS_WIDTH - w, y=2 * config.BORDERS_WIDTH + h + 10,
              width=w,
              height=screen_height - 3 * config.BORDERS_WIDTH - h - int(h / 3) + 50)
    lbl['borderwidth'] = 10
    lbl['relief'] = 'flat'

    lbl_static_text = tkinter.StringVar()
    lbl_static = tkinter.Label(root, textvariable=lbl_static_text, relief=tkinter.GROOVE,
                               bg=config.SCREEN_BG_COLOR, fg='white',
                               font='Arial 20 italic', bd=0,
                               justify=tkinter.LEFT)
    lbl_static_text.set('SF & misalignment errors are not shown.')
    w = config.VIDEO_PANEL_SIZE[0] * screen_width
    h = config.VIDEO_PANEL_SIZE[1] * screen_height
    lbl_static.place(x=screen_width - config.BORDERS_WIDTH - w,
                     y=screen_height - config.BORDERS_WIDTH + 60 - int(h / 3),
                     width=w,
                     height=int(h / 3) - 20)
    lbl_static['borderwidth'] = 10
    lbl_static['relief'] = 'flat'

    w = screen_width * config.PLOT_PANEL_SIZE[0]
    h = screen_height * config.PLOT_PANEL_SIZE[1]

    next_button = tkinter.Button(root, text='Next', fg='black',
                                 command=start_record_point, bd=5, font='Arial 28 bold')
    next_button.place(x=config.BORDERS_WIDTH, y=2 * config.BORDERS_WIDTH + h + 50,
                      height=screen_height - 3 * config.BORDERS_WIDTH - h - 50,
                      width=int(w / 3))

    camera_panel = None
    polar_view = tkinter.IntVar()
    polar_chack_box = tkinter.Checkbutton(root, text='Polar View', variable=polar_view, font='Arial 24 italic',
                                          bg=config.PLOT_BG_COLOR)
    polar_chack_box.place(x=config.BORDERS_WIDTH, y=2 * config.BORDERS_WIDTH + h + 10,
                          height=40, width=int(w / 3))

    fig = Figure(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_facecolor(config.PLOT_BG_COLOR)
    f_plot = fig.add_subplot(111)
    config_plot_style(f_plot)

    canvas = FigureCanvasTkAgg(fig, root)
    canvas.get_tk_widget()['relief'] = 'flat'
    canvas.get_tk_widget()['borderwidth'] = 10
    canvas.get_tk_widget().place(x=config.BORDERS_WIDTH, y=config.BORDERS_WIDTH, bordermode=tkinter.INSIDE)
    canvas.draw()

    data_source_proc = mp.Process(target=data_source, args=(stop_event, data_queue))
    data_source_proc.start()

    root.protocol("WM_DELETE_WINDOW", close_callback)
    root.after(ms=100, func=lambda: draw_data(canvas, f_plot, camera_panel, data_queue))

    root.mainloop()
