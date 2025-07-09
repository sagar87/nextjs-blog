import Link from "next/link";
import { formatDate } from "date-fns";
import { PostMetaData } from "./PostMetaData";

const PostPreview = (props: PostMetaData) => {
  return (
    <div
      className="border border-violet-200 p-5 rounded-md shadow-md bg-white"
      key={props.slug}
    >
      <Link href={`posts/${props.slug}`}>
        <h2 className="font-bold text-violet-500 hover:underline">
          {props.title}
        </h2>
      </Link>
      <p className="text-sm text-slate-400">
        {formatDate(props.date, "yyyy-MM-dd")}
      </p>
      <p className="text-slate-700">{props.subtitle}</p>
    </div>
  );
};

export default PostPreview;
