import xlwt

def write_x_excel(noise, name):
    num = len(noise[0])
    sample_size = len(noise)

    f = xlwt.Workbook()
    sheet1 = f.add_sheet('noise',cell_overwrite_ok=True)

    for i in range(sample_size):
        for j in range(num):
            sheet1.write(i, j, noise[i][j])

    f.save(name + '.xls')


def write_res_excel(name, eva_proposed, eva_PL, time_proposed, time_PL, error_count_proposed, error_count_PL):
    num = len(eva_proposed[0])
    # sample_size = len(noise)

    f = xlwt.Workbook()
    sheet1 = f.add_sheet('Eva_propsed',cell_overwrite_ok=True)
    sheet2 = f.add_sheet('Eva_PL',cell_overwrite_ok=True)
    sheet3 = f.add_sheet('Time',cell_overwrite_ok=True)

    for i in range(num):
        for j in range(num):
            sheet1.write(i, j, eva_proposed[i][j])

    for i in range(num):
        for j in range(num):
            sheet2.write(i, j, eva_PL[i][j])

    sheet3.write(0, 0, time_proposed)
    sheet3.write(1, 0, time_PL)
    sheet3.write(0, 1, error_count_proposed)
    sheet3.write(1, 1, error_count_PL)
    f.save(name + '.xls')