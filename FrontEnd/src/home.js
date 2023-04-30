import { useState, useEffect } from "react";
import { makeStyles, withStyles } from "@material-ui/core/styles";
import AppBar from "@material-ui/core/AppBar";
import Toolbar from "@material-ui/core/Toolbar";
import Typography from "@material-ui/core/Typography";
import Avatar from "@material-ui/core/Avatar";
import Container from "@material-ui/core/Container";
import React from "react";
import Card from "@material-ui/core/Card";
import CardContent from "@material-ui/core/CardContent";
import { Paper, CardActionArea, CardMedia, Grid, TableContainer, Table, TableBody, TableHead, TableRow, TableCell, Button, CircularProgress } from "@material-ui/core";
import logo from "./logo.png";
import image1 from "./bg.png";
import image from "./bg_green_scan.jpg"
import { DropzoneArea } from 'material-ui-dropzone';
import { common } from '@material-ui/core/colors';
import Clear from '@material-ui/icons/Clear';




const ColorButton = withStyles((theme) => ({
  root: {
    color: theme.palette.getContrastText(common.white),
    backgroundColor: common.white,
    '&:hover': {
      backgroundColor: '#ffffff7a',
    },
  },
}))(Button);
const axios = require("axios").default;

const useStyles = makeStyles((theme) => ({
  grow: {
    flexGrow: 1,
  },
  clearButton: {
    width: "-webkit-fill-available",
    borderRadius: "15px",
    padding: "15px 22px",
    color: "#000000a6",
    fontSize: "20px",
    fontWeight: 900,
  },
  root: {
    maxWidth: 345,
    flexGrow: 1,
  },
  media: {
    height: 400,
  },
  paper: {
    padding: theme.spacing(2),
    margin: 'auto',
    maxWidth: 500,
  },
  gridContainer: {
    justifyContent: "center",
    padding: "4em 1em 0 1em",
  },
  mainContainer: {
    backgroundImage: `url(${image})`,
    backgroundRepeat: 'no-repeat',
    backgroundPosition: 'center',
    backgroundSize: 'cover',
    height: "1080px",
    width: 'auto',
  },
  imageCard: {
    margin: "auto",
    maxWidth: 400,
    height: 500,
    backgroundColor: 'transparent',
    boxShadow: '0px 9px 70px 0px rgb(0 0 0 / 30%) !important',
    borderRadius: '15px',
  },
  imageCardEmpty: {
    height: 'auto',
  },
  noImage: {
    margin: "auto",
    width: 400,
    height: "400 !important",
  },
  input: {
    display: 'none',
  },
  uploadIcon: {
    background: 'white',
  },
  tableContainer: {
    backgroundColor: 'transparent !important',
    boxShadow: 'none !important',
  },
  table: {
    backgroundColor: 'transparent !important',
  },
  tableHead: {
    backgroundColor: 'transparent !important',
  },
  tableRow: {
    backgroundColor: 'transparent !important',
  },
  tableCell: {
    fontSize: '22px',
    backgroundColor: 'transparent !important',
    borderColor: 'transparent !important',
    color: '#000000a6 !important',
    fontWeight: 'bolder',
    padding: '1px 24px 1px 16px',
  },
  tableCellHead: {
    fontSize: '30px',
    backgroundColor: 'transparent !important',
    borderColor: 'transparent !important',
    color: '#000000a6 !important',
    fontWeight: 'bolder',
    padding: '1px 24px 1px 16px',
  },
  tableCell1: {
    fontSize: '14px',
    backgroundColor: 'transparent !important',
    borderColor: 'transparent !important',
    color: '#000000a6 !important',
    fontWeight: 'bolder',
    padding: '1px 24px 1px 16px',
  },
  tableBody: {
    backgroundColor: 'transparent !important',
  },
  text: {
    color: 'white !important',
    textAlign: 'center',
  },
  buttonGrid: {
    maxWidth: "416px",
    width: "100%",
  },
  label_detail: {
    backgroundColor: 'white',
    display: 'flex',
    justifyContent: 'center',
    flexDirection: 'column',
    alignItems: 'center',  
  },
  detail: {
    backgroundColor: 'white',
    display: 'flex',
    justifyContent: 'center',
    flexDirection: 'column',
    alignItems: 'center',
    borderRadius: '15px'
  },
  appbar: {
    backgroundColor: '#031b0f99 !important',
    color: '#25995c',
    boxShadow: 'none',
    fontFamily: 'Livvic',
    overflow: 'hidden',
    position: 'sticky',
    top: 0,
    left: 'auto',
    right: 0,
    fontSize: '2.25 rem'
  },
  loader: {
    color: '#be6a77 !important',
  }
}));
export const ImageUpload = () => {
  const classes = useStyles();
  const [selectedFile, setSelectedFile] = useState();
  const [preview, setPreview] = useState();
  const [data, setData] = useState();
  const [image, setImage] = useState(false);
  const [isLoading, setIsloading] = useState(false);
  let confidence = 0;

  const sendFile = async () => {
    if (image) {
      let formData = new FormData();
      formData.append("file", selectedFile);
      let res = await axios({
        method: "post",
        url: process.env.REACT_APP_API_URL,
        data: formData,
      });
      if (res.status === 200) {
        setData(res.data);
      }
      setIsloading(false);
    }
  }

  const clearData = () => {
    setData(null);
    setImage(false);
    setSelectedFile(null);
    setPreview(null);
  };

  useEffect(() => {
    if (!selectedFile) {
      setPreview(undefined);
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
  }, [selectedFile]);

  useEffect(() => {
    if (!preview) {
      return;
    }
    setIsloading(true);
    sendFile();
  }, [preview]);

  const onSelectFile = (files) => {
    if (!files || files.length === 0) {
      setSelectedFile(undefined);
      setImage(false);
      setData(undefined);
      return;
    }
    setSelectedFile(files[0]);
    setData(undefined);
    setImage(true);
  };

  if (data) {
    confidence = (parseFloat(data.confidence) * 100).toFixed(2);
  }

  return (
    <React.Fragment>
      <Container maxWidth={false} className={classes.mainContainer} disableGutters={true}>
        <AppBar position="static" className={classes.appbar}>
          <Toolbar>
            <Typography className={classes.title} variant="h6" noWrap>
              GreenScan
            </Typography>
            <div className={classes.grow} />
            <Avatar src={logo}></Avatar>
          </Toolbar>
        </AppBar>

        <Grid
          className={classes.gridContainer}
          container
          direction="row"
          justifyContent="center"
          alignItems="center"
          spacing={2}
        >

          {!data && <Grid item xs={12}>
            <Card className={`${classes.imageCard} ${!image ? classes.imageCardEmpty : ''}`}>
              {image && <CardActionArea>
                <CardMedia
                  className={classes.media}
                  image={preview}
                  component="image"
                  title="Contemplative Reptile"
                />
              </CardActionArea>
              }
              {!image && <CardContent className={classes.content}>
                <DropzoneArea
                  acceptedFiles={['image/*']}
                  dropzoneText={"Drag and drop an image of a plant leaf to process"}
                  onChange={onSelectFile}
                />
              </CardContent>}

              {isLoading && <CardContent className={classes.label_detail}>
                <CircularProgress color="secondary" className={classes.loader} />
                <Typography className={classes.title} variant="h6" noWrap>
                  Scanning
                </Typography>
              </CardContent>}
            </Card>
          </Grid>}

          {data && <Grid item xs={4}>
            <Card className={`${classes.imageCard} ${!image ? classes.imageCardEmpty : ''}`}>
              {image && <CardActionArea>
                <CardMedia
                  className={classes.media}
                  image={preview}
                  component="image"
                  title="Contemplative Reptile"
                />
              </CardActionArea>
              }
              {!image && <CardContent className={classes.content}>
                <DropzoneArea
                  acceptedFiles={['image/*']}
                  dropzoneText={"Drag and drop an image of a plant leaf to process"}
                  onChange={onSelectFile}
                />
              </CardContent>}

              {data && <CardContent className={classes.label_detail}>
                <TableContainer component={Paper} className={classes.tableContainer}>
                  <Table className={classes.table} size="small" aria-label="simple table">
                    <TableHead className={classes.tableHead}>
                      <TableRow className={classes.tableRow}>
                        <TableCell className={classes.tableCell1}>Label:</TableCell>
                        <TableCell align="right" className={classes.tableCell1}>Confidence:</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody className={classes.tableBody}>
                      <TableRow className={classes.tableRow}>
                        <TableCell component="th" scope="row" className={classes.tableCell}>
                          {data.class}
                        </TableCell>
                        <TableCell align="right" className={classes.tableCell}>{confidence}%</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>}
              {isLoading && <CardContent className={classes.detail}>
                <CircularProgress color="secondary" className={classes.loader} />
                <Typography className={classes.title} variant="h6" noWrap>
                  Scanning
                </Typography>
              </CardContent>}
            </Card>
          </Grid>}

          {data && <Grid item xs={8}>
            <CardContent className={classes.detail}>
              <TableContainer component={Paper} className={classes.tableContainer}>
                <Table className={classes.table} size="big" aria-label="simple table">
                  <TableBody className={classes.tableBody}>
                    <TableRow className={classes.tableRow}>
                      <TableCell component="th" scope="row" className={classes.tableCellHead}>{data.class}</TableCell>
                      {/* <TableCell align="right" className={classes.tableCell}>{confidence}%</TableCell> */}
                    </TableRow>
                    <TableRow className={classes.tableRow}>
                      <TableCell component="th" scope="row" className={classes.tableCell1}>{data.description}</TableCell>
                      <TableCell component="th" scope="row" className={classes.tableCell1}>
                        <img alt = {data.class} width={"250px"} src = {data.image_url} />
                      </TableCell>
                    </TableRow>
                    <TableRow className={classes.tableRow}>
                      <TableCell component="th" scope="row" className={classes.tableCell}>Recommended Steps:</TableCell>
                    </TableRow>
                    <TableRow className={classes.tableRow}>
                      <TableCell component="th" scope="row" className={classes.tableCell1}>{data.poss_steps}</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
                <Table className={classes.table} size="big" aria-label="simple table">
                    <TableRow className={classes.tableRow}>
                      <TableCell component="th" scope="row" className={classes.tableCell}>Supplement</TableCell>
                    </TableRow>
                    <TableRow className={classes.tableRow}>
                      <TableCell component="th" scope="row" className={classes.tableCell}>{data.supplement_name}</TableCell>
                      <TableCell component="th" scope="row" className={classes.tableCell1}>
                        <img alt = {data.class} width={"100px"} src = {data.supplement_img} />
                      </TableCell>
                      <TableCell component="th" scope="row" className={classes.tableCell1}>
                        <a href = {data.buy_link}>BUY THIS SUPPLEMENT</a>
                      </TableCell>
                    </TableRow>
                </Table>
              </TableContainer>
            </CardContent>
          </Grid>}

          {/* CLEAR BUTTON STARTS HERE */}
          {data &&
            <Grid item xs={12} className={classes.buttonGrid} >

              <ColorButton variant="contained" className={classes.clearButton} color="primary" component="span" size="large" onClick={clearData} startIcon={<Clear fontSize="large" />}>
                Clear
              </ColorButton>
            </Grid>}
          {/* CLEAR BUTTON ENDS HERE */}
        </Grid >

      </Container >
    </React.Fragment >
  );
};
