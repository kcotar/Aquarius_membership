import matplotlib
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk


class TkWindow:
    def __init__(self, w=500, h=500, title=''):
        # stream parameters
        self.ra_stream = 0.
        self.dec_stream = 0.
        self.rv_stream = 25.

        # window
        self.main_window = tk.Tk()
        self.main_window.title(title)
        canvas = tk.Canvas(self.main_window, width=w, height=h)
        canvas.pack()

        # create first input rows
        x_off = 5
        y_line_w = 30
        l_title = tk.Label(self.main_window, text='Initial input stream parameters')
        l_title.place(x=x_off, y=x_off)
        # RA
        l_ra = tk.Label(self.main_window, text='RA: ')
        l_ra.place(x=x_off, y=y_line_w+x_off)
        self.ra_entry = tk.Entry(self.main_window, width=8)
        self.ra_entry.place(x=x_off+30, y=y_line_w+x_off)
        ra_p = tk.Button(self.main_window, text='+', command=lambda: self.ra_set(sign='+'))
        ra_p.place(x=x_off+100, y=y_line_w+x_off, width=30, height=20)
        ra_m = tk.Button(self.main_window, text='-', command=lambda: self.ra_set(sign='-'))
        ra_m.place(x=x_off+130, y=y_line_w + x_off, width=30, height=20)
        # DEC
        l_dec = tk.Label(self.main_window, text='DEC: ')
        l_dec.place(x=x_off, y=2*y_line_w + x_off)
        self.dec_entry = tk.Entry(self.main_window, width=8)
        self.dec_entry.place(x=x_off + 30, y=2*y_line_w + x_off)
        dec_p = tk.Button(self.main_window, text='+', command=lambda: self.dec_set(sign='+'))
        dec_p.place(x=x_off + 100, y=2*y_line_w + x_off, width=30, height=20)
        dec_m = tk.Button(self.main_window, text='-', command=lambda: self.dec_set(sign='-'))
        dec_m.place(x=x_off + 130, y=2*y_line_w + x_off, width=30, height=20)
        # RV
        l_rv = tk.Label(self.main_window, text='RV: ')
        l_rv.place(x=x_off, y=3 * y_line_w + x_off)
        self.rv_entry = tk.Entry(self.main_window, width=8)
        self.rv_entry.place(x=x_off + 30, y=3 * y_line_w + x_off)
        rv_p = tk.Button(self.main_window, text='+', command=lambda: self.rv_set(sign='+'))
        rv_p.place(x=x_off + 100, y=3 * y_line_w + x_off, width=30, height=20)
        rv_m = tk.Button(self.main_window, text='-', command=lambda: self.rv_set(sign='-'))
        rv_m.place(x=x_off + 130, y=3 * y_line_w + x_off, width=30, height=20)
        # show values
        self.update_values()
        # large plot button
        plot_b = tk.Button(self.main_window, text='Begin stream\n\nanalysis', command=lambda: self.plot_selected())
        plot_b.place(x=200, y=35, width=120, height=80)

    def ra_set(self, sign='+'):
        if sign is '+':
            self.ra_stream += 5.
        elif sign is '-':
            self.ra_stream -= 5.
        self.update_values()

    def dec_set(self, sign='+'):
        if sign is '+':
            self.dec_stream += 5.
        elif sign is '-':
            self.dec_stream -= 5.
        self.update_values()

    def rv_set(self, sign='+'):
        if sign is '+':
            self.rv_stream += 5.
        elif sign is '-':
            self.rv_stream -= 5.
        self.update_values()

    def update_values(self):
        self.ra_entry.delete(0, tk.END)
        self.ra_entry.insert(0, str(self.ra_stream))
        self.dec_entry.delete(0, tk.END)
        self.dec_entry.insert(0, str(self.dec_stream))
        self.rv_entry.delete(0, tk.END)
        self.rv_entry.insert(0, str(self.rv_stream))

    def plot_selected(self):
        d=1

    def start(self):
        tk.mainloop()
